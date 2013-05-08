//===-- nullptr-convert/NullptrActions.cpp - Matcher callback -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the definition of the NullptrFixer class which is
///  used as an ASTMatcher callback. Also within this file is a helper AST
///  visitor class used to identify sequences of explicit casts.
///
//===----------------------------------------------------------------------===//

#include "NullptrActions.h"
#include "NullptrMatchers.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "clang/Basic/CharInfo.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace clang;

namespace {

const char *NullMacroName = "NULL";

static llvm::cl::opt<std::string> UserNullMacroNames(
    "user-null-macros", llvm::cl::desc("Comma-separated list of user-defined "
                                       "macro names that behave like NULL"),
    llvm::cl::init(""));

bool isReplaceableRange(SourceLocation StartLoc, SourceLocation EndLoc,
                        const SourceManager &SM) {
  return SM.isFromSameFile(StartLoc, EndLoc) && SM.isFromMainFile(StartLoc);
}

/// \brief Replaces the provided range with the text "nullptr", but only if
/// the start and end location are both in main file.
/// Returns true if and only if a replacement was made.
void ReplaceWithNullptr(tooling::Replacements &Replace, SourceManager &SM,
                        SourceLocation StartLoc, SourceLocation EndLoc) {
  CharSourceRange Range(SourceRange(StartLoc, EndLoc), true);
  // Add a space if nullptr follows an alphanumeric character. This happens
  // whenever there is an c-style explicit cast to nullptr not surrounded by
  // parentheses and right beside a return statement.
  SourceLocation PreviousLocation = StartLoc.getLocWithOffset(-1);
  if (isAlphanumeric(*FullSourceLoc(PreviousLocation, SM).getCharacterData()))
    Replace.insert(tooling::Replacement(SM, Range, " nullptr"));
  else
    Replace.insert(tooling::Replacement(SM, Range, "nullptr"));
}

/// \brief Returns the name of the outermost macro.
///
/// Given
/// \code
/// #define MY_NULL NULL
/// \endcode
/// If \p Loc points to NULL, this function will return the name MY_NULL.
llvm::StringRef GetOutermostMacroName(
    SourceLocation Loc, const SourceManager &SM, const LangOptions &LO) {
  assert(Loc.isMacroID());
  SourceLocation OutermostMacroLoc;

  while (Loc.isMacroID()) {
    OutermostMacroLoc = Loc;
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }

  return clang::Lexer::getImmediateMacroName(OutermostMacroLoc, SM, LO);
}

/// \brief Given the SourceLocation for a macro arg expansion, finds the
/// non-macro SourceLocation of the macro the arg was passed to and the
/// non-macro SourceLocation of the argument in the arg list to that macro.
/// These results are returned via \c MacroLoc and \c ArgLoc respectively.
/// These values are undefined if the return value is false.
///
/// \returns false if one of the returned SourceLocations would be a
/// SourceLocation pointing within the definition of another macro.
bool getMacroAndArgLocations(SourceLocation Loc, const SourceManager &SM,
                             SourceLocation &ArgLoc, SourceLocation &MacroLoc) {
  assert(Loc.isMacroID() && "Only reasonble to call this on macros");

  ArgLoc = Loc;

  // Find the location of the immediate macro expansion.
  while (1) {
    std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(ArgLoc);
    const SrcMgr::SLocEntry *E = &SM.getSLocEntry(LocInfo.first);
    const SrcMgr::ExpansionInfo &Expansion = E->getExpansion();

    ArgLoc = Expansion.getExpansionLocStart();
    if (!Expansion.isMacroArgExpansion()) {
      // TODO: Insert test for user-defined null macro here.
      return MacroLoc.isFileID();
    }

    MacroLoc = SM.getImmediateExpansionRange(ArgLoc).first;

    ArgLoc = Expansion.getSpellingLoc().getLocWithOffset(LocInfo.second);
    if (ArgLoc.isFileID())
      return true;

    // If spelling location resides in the same FileID as macro expansion
    // location, it means there is no inner macro.
    FileID MacroFID = SM.getFileID(MacroLoc);
    if (SM.isInFileID(ArgLoc, MacroFID))
      // Don't transform this case. If the characters that caused the
      // null-conversion come from within a macro, they can't be changed.
      return false;
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
/// - TestLoc is not from a macro expansion
/// - TestLoc is from a different macro expansion
bool expandsFrom(SourceLocation TestLoc, SourceLocation TestMacroLoc,
                 const SourceManager &SM) {
  if (TestLoc.isFileID()) {
    return false;
  }

  SourceLocation Loc = TestLoc, MacroLoc;

  while (1) {
    std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
    const SrcMgr::SLocEntry *E = &SM.getSLocEntry(LocInfo.first);
    const SrcMgr::ExpansionInfo &Expansion = E->getExpansion();

    Loc = Expansion.getExpansionLocStart();

    if (!Expansion.isMacroArgExpansion()) {
      if (Loc.isFileID()) {
        if (Loc == TestMacroLoc)
          // Match made.
          return true;
        return false;
      }
      // Since Loc is still a macro ID and it's not an argument expansion, we
      // don't need to do the work of handling an argument expansion. Simply
      // keep recursively expanding until we hit a FileID or a macro arg
      // expansion or a macro arg expansion.
      continue;
    }

    MacroLoc = SM.getImmediateExpansionRange(Loc).first;
    if (MacroLoc.isFileID() && MacroLoc == TestMacroLoc)
      // Match made.
      return true;

    Loc = Expansion.getSpellingLoc();
    Loc = Expansion.getSpellingLoc().getLocWithOffset(LocInfo.second);
    if (Loc.isFileID())
      // If we made it this far without finding a match, there is no match to
      // be made.
      return false;
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
                            ast_type_traits::DynTypedNode &Result,
                            ASTContext &Context) {
  // Below we're only following the first parent back up the AST. This should
  // be fine since for the statements we care about there should only be one
  // parent as far up as we care. If this assumption doesn't hold, need to
  // revisit what to do here.

  assert(MacroLoc.isFileID());

  do {
    ASTContext::ParentVector Parents = Context.getParents(Start);
    if (Parents.empty())
      return false;
    assert(Parents.size() == 1 &&
           "Found an ancestor with more than one parent!");

    ASTContext::ParentVector::const_iterator I = Parents.begin();

    SourceLocation Loc;
    if (const Decl *D = I->get<Decl>())
      Loc = D->getLocStart();
    else if (const Stmt *S = I->get<Stmt>())
      Loc = S->getLocStart();
    else
      llvm_unreachable("Expected to find Decl or Stmt containing ancestor");

    if (!expandsFrom(Loc, MacroLoc, Context.getSourceManager())) {
      Result = *I;
      return true;
    }
    Start = *I;
  } while(1);

  llvm_unreachable("findContainingAncestor");
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
    if (SM.getFileLoc(S->getLocStart()) != CastLoc)
      return true;
    Visited = true;

    const ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(S);
    if (Cast && (Cast->getCastKind() == CK_NullToPointer ||
                 Cast->getCastKind() == CK_NullToMemberPointer))
      CastFound = true;

    return true;
  }

  bool foundInvalid() const { return InvalidFound; }

private:
  SourceLocation CastLoc;
  const SourceManager &SM;

  bool Visited;
  bool CastFound;
  bool InvalidFound;
};

/// \brief Tests that all expansions of a macro arg, one of which expands to
/// result in \p CE, yield NullTo(Member)Pointer casts.
bool allArgUsesValid(const CastExpr *CE, ASTContext &Context) {
  SourceLocation CastLoc = CE->getLocStart();
  const SourceManager &SM = Context.getSourceManager();

  // Step 1: Get location of macro arg and location of the macro the arg was
  // provided to.
  SourceLocation ArgLoc, MacroLoc;
  if (!getMacroAndArgLocations(CastLoc, SM, ArgLoc, MacroLoc))
    return false;

  // Step 2: Find the first ancestor that doesn't expand from this macro.
  ast_type_traits::DynTypedNode ContainingAncestor;
  if (!findContainingAncestor(ast_type_traits::DynTypedNode::create<Stmt>(*CE),
                              MacroLoc, ContainingAncestor, Context))
    return false;

  // Step 3:
  // Visit children of this containing parent looking for the least-descended
  // nodes of the containing parent which are macro arg expansions that expand
  // from the given arg location.
  // Visitor needs: arg loc
  MacroArgUsageVisitor ArgUsageVisitor(SM.getFileLoc(CastLoc), SM);
  if (const Decl *D = ContainingAncestor.get<Decl>())
    ArgUsageVisitor.TraverseDecl(const_cast<Decl*>(D));
  else if (const Stmt *S = ContainingAncestor.get<Stmt>())
    ArgUsageVisitor.TraverseStmt(const_cast<Stmt*>(S));
  else
    llvm_unreachable("Unhandled ContainingAncestor node type");

  if (ArgUsageVisitor.foundInvalid())
    return false;

  return true;
}

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
  CastSequenceVisitor(tooling::Replacements &R, ASTContext &Context,
                      const UserMacroNames &UserNullMacros,
                      unsigned &AcceptedChanges)
      : Replace(R), SM(Context.getSourceManager()), Context(Context),
        UserNullMacros(UserNullMacros), AcceptedChanges(AcceptedChanges),
        FirstSubExpr(0), PruneSubtree(false) {}

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
    CastExpr *C = dyn_cast<CastExpr>(S);
    if (!C) {
      FirstSubExpr = 0;
      return true;
    } else if (!FirstSubExpr) {
      FirstSubExpr = C->getSubExpr()->IgnoreParens();
    }

    if (C->getCastKind() == CK_NullToPointer ||
        C->getCastKind() == CK_NullToMemberPointer) {

      SourceLocation StartLoc = FirstSubExpr->getLocStart();
      SourceLocation EndLoc = FirstSubExpr->getLocEnd();

      // If the location comes from a macro arg expansion, *all* uses of that
      // arg must be checked to result in NullTo(Member)Pointer casts.
      //
      // If the location comes from a macro body expansion, check to see if its
      // coming from one of the allowed 'NULL' macros.
      if (SM.isMacroArgExpansion(StartLoc) && SM.isMacroArgExpansion(EndLoc)) {
        SourceLocation FileLocStart = SM.getFileLoc(StartLoc),
                       FileLocEnd = SM.getFileLoc(EndLoc);
        if (isReplaceableRange(FileLocStart, FileLocEnd, SM) &&
            allArgUsesValid(C, Context)) {
          ReplaceWithNullptr(Replace, SM, FileLocStart, FileLocEnd);
          ++AcceptedChanges;
        }
        return skipSubTree();
      }

      if (SM.isMacroBodyExpansion(StartLoc) &&
          SM.isMacroBodyExpansion(EndLoc)) {
        llvm::StringRef OutermostMacroName =
            GetOutermostMacroName(StartLoc, SM, Context.getLangOpts());

        // Check to see if the user wants to replace the macro being expanded.
        if (std::find(UserNullMacros.begin(), UserNullMacros.end(),
                      OutermostMacroName) == UserNullMacros.end()) {
          return skipSubTree();
        }

        StartLoc = SM.getFileLoc(StartLoc);
        EndLoc = SM.getFileLoc(EndLoc);
      }

      if (!isReplaceableRange(StartLoc, EndLoc, SM)) {
        return skipSubTree();
      }
      ReplaceWithNullptr(Replace, SM, StartLoc, EndLoc);
      ++AcceptedChanges;

      return skipSubTree();
    } // If NullTo(Member)Pointer cast.

    return true;
  }

private:
  bool skipSubTree() { PruneSubtree = true; return true; }

private:
  tooling::Replacements &Replace;
  SourceManager &SM;
  ASTContext &Context;
  const UserMacroNames &UserNullMacros;
  unsigned &AcceptedChanges;
  Expr *FirstSubExpr;
  bool PruneSubtree;
};
} // namespace

NullptrFixer::NullptrFixer(clang::tooling::Replacements &Replace,
                           unsigned &AcceptedChanges, RiskLevel)
    : Replace(Replace), AcceptedChanges(AcceptedChanges) {
  if (!UserNullMacroNames.empty()) {
    llvm::StringRef S = UserNullMacroNames;
    S.split(UserNullMacros, ",");
  }
  UserNullMacros.insert(UserNullMacros.begin(), llvm::StringRef(NullMacroName));
}

void NullptrFixer::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  const CastExpr *NullCast = Result.Nodes.getNodeAs<CastExpr>(CastSequence);
  assert(NullCast && "Bad Callback. No node provided");
  // Given an implicit null-ptr cast or an explicit cast with an implicit
  // null-to-pointer cast within use CastSequenceVisitor to identify sequences
  // of explicit casts that can be converted into 'nullptr'.
  CastSequenceVisitor Visitor(Replace, *Result.Context, UserNullMacros,
                              AcceptedChanges);
  Visitor.TraverseStmt(const_cast<CastExpr *>(NullCast));
}
