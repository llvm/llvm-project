//===--- UseNullptrCheck.cpp - clang-tidy----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseNullptrCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;

namespace clang {
namespace tidy {
namespace modernize {
namespace {

const char CastSequence[] = "sequence";

AST_MATCHER(Type, sugaredNullptrType) {
  const Type *DesugaredType = Node.getUnqualifiedDesugaredType();
  if (const auto *BT = dyn_cast<BuiltinType>(DesugaredType))
    return BT->getKind() == BuiltinType::NullPtr;
  return false;
}

/// \brief Create a matcher that finds implicit casts as well as the head of a
/// sequence of zero or more nested explicit casts that have an implicit cast
/// to null within.
/// Finding sequences of explict casts is necessary so that an entire sequence
/// can be replaced instead of just the inner-most implicit cast.
StatementMatcher makeCastSequenceMatcher() {
  StatementMatcher ImplicitCastToNull = implicitCastExpr(
      anyOf(hasCastKind(CK_NullToPointer), hasCastKind(CK_NullToMemberPointer)),
      unless(hasImplicitDestinationType(qualType(substTemplateTypeParmType()))),
      unless(hasSourceExpression(hasType(sugaredNullptrType()))));

  return castExpr(anyOf(ImplicitCastToNull,
                        explicitCastExpr(hasDescendant(ImplicitCastToNull))),
                  unless(hasAncestor(explicitCastExpr())))
      .bind(CastSequence);
}

bool isReplaceableRange(SourceLocation StartLoc, SourceLocation EndLoc,
                        const SourceManager &SM) {
  return SM.isWrittenInSameFile(StartLoc, EndLoc);
}

/// \brief Replaces the provided range with the text "nullptr", but only if
/// the start and end location are both in main file.
/// Returns true if and only if a replacement was made.
void replaceWithNullptr(ClangTidyCheck &Check, SourceManager &SM,
                        SourceLocation StartLoc, SourceLocation EndLoc) {
  CharSourceRange Range(SourceRange(StartLoc, EndLoc), true);
  // Add a space if nullptr follows an alphanumeric character. This happens
  // whenever there is an c-style explicit cast to nullptr not surrounded by
  // parentheses and right beside a return statement.
  SourceLocation PreviousLocation = StartLoc.getLocWithOffset(-1);
  bool NeedsSpace = isAlphanumeric(*SM.getCharacterData(PreviousLocation));
  Check.diag(Range.getBegin(), "use nullptr") << FixItHint::CreateReplacement(
      Range, NeedsSpace ? " nullptr" : "nullptr");
}

/// \brief Returns the name of the outermost macro.
///
/// Given
/// \code
/// #define MY_NULL NULL
/// \endcode
/// If \p Loc points to NULL, this function will return the name MY_NULL.
StringRef getOutermostMacroName(SourceLocation Loc, const SourceManager &SM,
                                const LangOptions &LO) {
  assert(Loc.isMacroID());
  SourceLocation OutermostMacroLoc;

  while (Loc.isMacroID()) {
    OutermostMacroLoc = Loc;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }

  return Lexer::getImmediateMacroName(OutermostMacroLoc, SM, LO);
}

/// \brief RecursiveASTVisitor for ensuring all nodes rooted at a given AST
/// subtree that have file-level source locations corresponding to a macro
/// argument have implicit NullTo(Member)Pointer nodes as ancestors.
class MacroArgUsageVisitor : public RecursiveASTVisitor<MacroArgUsageVisitor> {
public:
  MacroArgUsageVisitor(SourceLocation CastLoc, const SourceManager &SM)
      : CastLoc(CastLoc), SM(SM), Visited(false), CastFound(false),
        InvalidFound(false) {
    assert(CastLoc.isFileID());
  }

  bool TraverseStmt(Stmt *S) {
    bool VisitedPreviously = Visited;

    if (!RecursiveASTVisitor<MacroArgUsageVisitor>::TraverseStmt(S))
      return false;

    // The point at which VisitedPreviously is false and Visited is true is the
    // root of a subtree containing nodes whose locations match CastLoc. It's
    // at this point we test that the Implicit NullTo(Member)Pointer cast was
    // found or not.
    if (!VisitedPreviously) {
      if (Visited && !CastFound) {
        // Found nodes with matching SourceLocations but didn't come across a
        // cast. This is an invalid macro arg use. Can stop traversal
        // completely now.
        InvalidFound = true;
        return false;
      }
      // Reset state as we unwind back up the tree.
      CastFound = false;
      Visited = false;
    }
    return true;
  }

  bool VisitStmt(Stmt *S) {
    if (SM.getFileLoc(S->getBeginLoc()) != CastLoc)
      return true;
    Visited = true;

    const ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(S);
    if (Cast && (Cast->getCastKind() == CK_NullToPointer ||
                 Cast->getCastKind() == CK_NullToMemberPointer))
      CastFound = true;

    return true;
  }

  bool TraverseInitListExpr(InitListExpr *S) {
    // Only go through the semantic form of the InitListExpr, because
    // ImplicitCast might not appear in the syntactic form, and this results in
    // finding usages of the macro argument that don't have a ImplicitCast as an
    // ancestor (thus invalidating the replacement) when they actually have.
    return RecursiveASTVisitor<MacroArgUsageVisitor>::
        TraverseSynOrSemInitListExpr(
            S->isSemanticForm() ? S : S->getSemanticForm());
  }

  bool foundInvalid() const { return InvalidFound; }

private:
  SourceLocation CastLoc;
  const SourceManager &SM;

  bool Visited;
  bool CastFound;
  bool InvalidFound;
};

/// \brief Looks for implicit casts as well as sequences of 0 or more explicit
/// casts with an implicit null-to-pointer cast within.
///
/// The matcher this visitor is used with will find a single implicit cast or a
/// top-most explicit cast (i.e. it has no explicit casts as an ancestor) where
/// an implicit cast is nested within. However, there is no guarantee that only
/// explicit casts exist between the found top-most explicit cast and the
/// possibly more than one nested implicit cast. This visitor finds all cast
/// sequences with an implicit cast to null within and creates a replacement
/// leaving the outermost explicit cast unchanged to avoid introducing
/// ambiguities.
class CastSequenceVisitor : public RecursiveASTVisitor<CastSequenceVisitor> {
public:
  CastSequenceVisitor(ASTContext &Context, ArrayRef<StringRef> NullMacros,
                      ClangTidyCheck &check)
      : SM(Context.getSourceManager()), Context(Context),
        NullMacros(NullMacros), Check(check), FirstSubExpr(nullptr),
        PruneSubtree(false) {}

  bool TraverseStmt(Stmt *S) {
    // Stop traversing down the tree if requested.
    if (PruneSubtree) {
      PruneSubtree = false;
      return true;
    }
    return RecursiveASTVisitor<CastSequenceVisitor>::TraverseStmt(S);
  }

  // Only VisitStmt is overridden as we shouldn't find other base AST types
  // within a cast expression.
  bool VisitStmt(Stmt *S) {
    auto *C = dyn_cast<CastExpr>(S);
    // Catch the castExpr inside cxxDefaultArgExpr.
    if (auto *E = dyn_cast<CXXDefaultArgExpr>(S)) {
      C = dyn_cast<CastExpr>(E->getExpr());
      FirstSubExpr = nullptr;
    }
    if (!C) {
      FirstSubExpr = nullptr;
      return true;
    }

    auto* CastSubExpr = C->getSubExpr()->IgnoreParens();
    // Ignore cast expressions which cast nullptr literal.
    if (isa<CXXNullPtrLiteralExpr>(CastSubExpr)) {
      return true;
    }

    if (!FirstSubExpr)
      FirstSubExpr = CastSubExpr;

    if (C->getCastKind() != CK_NullToPointer &&
        C->getCastKind() != CK_NullToMemberPointer) {
      return true;
    }

    SourceLocation StartLoc = FirstSubExpr->getBeginLoc();
    SourceLocation EndLoc = FirstSubExpr->getEndLoc();

    // If the location comes from a macro arg expansion, *all* uses of that
    // arg must be checked to result in NullTo(Member)Pointer casts.
    //
    // If the location comes from a macro body expansion, check to see if its
    // coming from one of the allowed 'NULL' macros.
    if (SM.isMacroArgExpansion(StartLoc) && SM.isMacroArgExpansion(EndLoc)) {
      SourceLocation FileLocStart = SM.getFileLoc(StartLoc),
                     FileLocEnd = SM.getFileLoc(EndLoc);
      SourceLocation ImmediateMacroArgLoc, MacroLoc;
      // Skip NULL macros used in macro.
      if (!getMacroAndArgLocations(StartLoc, ImmediateMacroArgLoc, MacroLoc) ||
          ImmediateMacroArgLoc != FileLocStart)
        return skipSubTree();

      if (isReplaceableRange(FileLocStart, FileLocEnd, SM) &&
          allArgUsesValid(C)) {
        replaceWithNullptr(Check, SM, FileLocStart, FileLocEnd);
      }
      return true;
    }

    if (SM.isMacroBodyExpansion(StartLoc) && SM.isMacroBodyExpansion(EndLoc)) {
      StringRef OutermostMacroName =
          getOutermostMacroName(StartLoc, SM, Context.getLangOpts());

      // Check to see if the user wants to replace the macro being expanded.
      if (std::find(NullMacros.begin(), NullMacros.end(), OutermostMacroName) ==
          NullMacros.end()) {
        return skipSubTree();
      }

      StartLoc = SM.getFileLoc(StartLoc);
      EndLoc = SM.getFileLoc(EndLoc);
    }

    if (!isReplaceableRange(StartLoc, EndLoc, SM)) {
      return skipSubTree();
    }
    replaceWithNullptr(Check, SM, StartLoc, EndLoc);

    return true;
  }

private:
  bool skipSubTree() {
    PruneSubtree = true;
    return true;
  }

  /// \brief Tests that all expansions of a macro arg, one of which expands to
  /// result in \p CE, yield NullTo(Member)Pointer casts.
  bool allArgUsesValid(const CastExpr *CE) {
    SourceLocation CastLoc = CE->getBeginLoc();

    // Step 1: Get location of macro arg and location of the macro the arg was
    // provided to.
    SourceLocation ArgLoc, MacroLoc;
    if (!getMacroAndArgLocations(CastLoc, ArgLoc, MacroLoc))
      return false;

    // Step 2: Find the first ancestor that doesn't expand from this macro.
    ast_type_traits::DynTypedNode ContainingAncestor;
    if (!findContainingAncestor(
            ast_type_traits::DynTypedNode::create<Stmt>(*CE), MacroLoc,
            ContainingAncestor))
      return false;

    // Step 3:
    // Visit children of this containing parent looking for the least-descended
    // nodes of the containing parent which are macro arg expansions that expand
    // from the given arg location.
    // Visitor needs: arg loc.
    MacroArgUsageVisitor ArgUsageVisitor(SM.getFileLoc(CastLoc), SM);
    if (const auto *D = ContainingAncestor.get<Decl>())
      ArgUsageVisitor.TraverseDecl(const_cast<Decl *>(D));
    else if (const auto *S = ContainingAncestor.get<Stmt>())
      ArgUsageVisitor.TraverseStmt(const_cast<Stmt *>(S));
    else
      llvm_unreachable("Unhandled ContainingAncestor node type");

    return !ArgUsageVisitor.foundInvalid();
  }

  /// \brief Given the SourceLocation for a macro arg expansion, finds the
  /// non-macro SourceLocation of the macro the arg was passed to and the
  /// non-macro SourceLocation of the argument in the arg list to that macro.
  /// These results are returned via \c MacroLoc and \c ArgLoc respectively.
  /// These values are undefined if the return value is false.
  ///
  /// \returns false if one of the returned SourceLocations would be a
  /// SourceLocation pointing within the definition of another macro.
  bool getMacroAndArgLocations(SourceLocation Loc, SourceLocation &ArgLoc,
                               SourceLocation &MacroLoc) {
    assert(Loc.isMacroID() && "Only reasonble to call this on macros");

    ArgLoc = Loc;

    // Find the location of the immediate macro expansion.
    while (true) {
      std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(ArgLoc);
      const SrcMgr::SLocEntry *E = &SM.getSLocEntry(LocInfo.first);
      const SrcMgr::ExpansionInfo &Expansion = E->getExpansion();

      SourceLocation OldArgLoc = ArgLoc;
      ArgLoc = Expansion.getExpansionLocStart();
      if (!Expansion.isMacroArgExpansion()) {
        if (!MacroLoc.isFileID())
          return false;

        StringRef Name =
            Lexer::getImmediateMacroName(OldArgLoc, SM, Context.getLangOpts());
        return std::find(NullMacros.begin(), NullMacros.end(), Name) !=
               NullMacros.end();
      }

      MacroLoc = SM.getExpansionRange(ArgLoc).getBegin();

      ArgLoc = Expansion.getSpellingLoc().getLocWithOffset(LocInfo.second);
      if (ArgLoc.isFileID())
        return true;

      // If spelling location resides in the same FileID as macro expansion
      // location, it means there is no inner macro.
      FileID MacroFID = SM.getFileID(MacroLoc);
      if (SM.isInFileID(ArgLoc, MacroFID)) {
        // Don't transform this case. If the characters that caused the
        // null-conversion come from within a macro, they can't be changed.
        return false;
      }
    }

    llvm_unreachable("getMacroAndArgLocations");
  }

  /// \brief Tests if TestMacroLoc is found while recursively unravelling
  /// expansions starting at TestLoc. TestMacroLoc.isFileID() must be true.
  /// Implementation is very similar to getMacroAndArgLocations() except in this
  /// case, it's not assumed that TestLoc is expanded from a macro argument.
  /// While unravelling expansions macro arguments are handled as with
  /// getMacroAndArgLocations() but in this function macro body expansions are
  /// also handled.
  ///
  /// False means either:
  /// - TestLoc is not from a macro expansion.
  /// - TestLoc is from a different macro expansion.
  bool expandsFrom(SourceLocation TestLoc, SourceLocation TestMacroLoc) {
    if (TestLoc.isFileID()) {
      return false;
    }

    SourceLocation Loc = TestLoc, MacroLoc;

    while (true) {
      std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
      const SrcMgr::SLocEntry *E = &SM.getSLocEntry(LocInfo.first);
      const SrcMgr::ExpansionInfo &Expansion = E->getExpansion();

      Loc = Expansion.getExpansionLocStart();

      if (!Expansion.isMacroArgExpansion()) {
        if (Loc.isFileID()) {
          return Loc == TestMacroLoc;
        }
        // Since Loc is still a macro ID and it's not an argument expansion, we
        // don't need to do the work of handling an argument expansion. Simply
        // keep recursively expanding until we hit a FileID or a macro arg
        // expansion or a macro arg expansion.
        continue;
      }

      MacroLoc = SM.getImmediateExpansionRange(Loc).getBegin();
      if (MacroLoc.isFileID() && MacroLoc == TestMacroLoc) {
        // Match made.
        return true;
      }

      Loc = Expansion.getSpellingLoc().getLocWithOffset(LocInfo.second);
      if (Loc.isFileID()) {
        // If we made it this far without finding a match, there is no match to
        // be made.
        return false;
      }
    }

    llvm_unreachable("expandsFrom");
  }

  /// \brief Given a starting point \c Start in the AST, find an ancestor that
  /// doesn't expand from the macro called at file location \c MacroLoc.
  ///
  /// \pre MacroLoc.isFileID()
  /// \returns true if such an ancestor was found, false otherwise.
  bool findContainingAncestor(ast_type_traits::DynTypedNode Start,
                              SourceLocation MacroLoc,
                              ast_type_traits::DynTypedNode &Result) {
    // Below we're only following the first parent back up the AST. This should
    // be fine since for the statements we care about there should only be one
    // parent, except for the case specified below.

    assert(MacroLoc.isFileID());

    while (true) {
      const auto &Parents = Context.getParents(Start);
      if (Parents.empty())
        return false;
      if (Parents.size() > 1) {
        // If there are more than one parents, don't do the replacement unless
        // they are InitListsExpr (semantic and syntactic form). In this case we
        // can choose any one here, and the ASTVisitor will take care of
        // traversing the right one.
        for (const auto &Parent : Parents) {
          if (!Parent.get<InitListExpr>())
            return false;
        }
      }

      const ast_type_traits::DynTypedNode &Parent = Parents[0];

      SourceLocation Loc;
      if (const auto *D = Parent.get<Decl>())
        Loc = D->getBeginLoc();
      else if (const auto *S = Parent.get<Stmt>())
        Loc = S->getBeginLoc();

      // TypeLoc and NestedNameSpecifierLoc are members of the parent map. Skip
      // them and keep going up.
      if (Loc.isValid()) {
        if (!expandsFrom(Loc, MacroLoc)) {
          Result = Parent;
          return true;
        }
      }
      Start = Parent;
    }

    llvm_unreachable("findContainingAncestor");
  }

private:
  SourceManager &SM;
  ASTContext &Context;
  ArrayRef<StringRef> NullMacros;
  ClangTidyCheck &Check;
  Expr *FirstSubExpr;
  bool PruneSubtree;
};

} // namespace

UseNullptrCheck::UseNullptrCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      NullMacrosStr(Options.get("NullMacros", "")) {
  StringRef(NullMacrosStr).split(NullMacros, ",");
}

void UseNullptrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "NullMacros", NullMacrosStr);
}

void UseNullptrCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matcher for C++. Because this checker is used for
  // modernization, it is reasonable to run it on any C++ standard with the
  // assumption the user is trying to modernize their codebase.
  if (getLangOpts().CPlusPlus)
    Finder->addMatcher(makeCastSequenceMatcher(), this);
}

void UseNullptrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *NullCast = Result.Nodes.getNodeAs<CastExpr>(CastSequence);
  assert(NullCast && "Bad Callback. No node provided");

  // Given an implicit null-ptr cast or an explicit cast with an implicit
  // null-to-pointer cast within use CastSequenceVisitor to identify sequences
  // of explicit casts that can be converted into 'nullptr'.
  CastSequenceVisitor(*Result.Context, NullMacros, *this)
      .TraverseStmt(const_cast<CastExpr *>(NullCast));
}

} // namespace modernize
} // namespace tidy
} // namespace clang
