//===-- tools/extra/clang-reorder-fields/ReorderFieldsAction.cpp -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the
/// ReorderFieldsAction::newASTConsumer method
///
//===----------------------------------------------------------------------===//

#include "ReorderFieldsAction.h"
#include "Designator.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Refactoring.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

namespace clang {
namespace reorder_fields {
using namespace clang::ast_matchers;
using llvm::SmallSetVector;

/// Finds the definition of a record by name.
///
/// \returns nullptr if the name is ambiguous or not found.
static const RecordDecl *findDefinition(StringRef RecordName,
                                        ASTContext &Context) {
  auto Results =
      match(recordDecl(hasName(RecordName), isDefinition()).bind("recordDecl"),
            Context);
  if (Results.empty()) {
    llvm::errs() << "Definition of " << RecordName << "  not found\n";
    return nullptr;
  }
  if (Results.size() > 1) {
    llvm::errs() << "The name " << RecordName
                 << " is ambiguous, several definitions found\n";
    return nullptr;
  }
  return selectFirst<RecordDecl>("recordDecl", Results);
}

static bool declaresMultipleFieldsInStatement(const RecordDecl *Decl) {
  SourceLocation LastTypeLoc;
  for (const auto &Field : Decl->fields()) {
    SourceLocation TypeLoc =
        Field->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
    if (LastTypeLoc.isValid() && TypeLoc == LastTypeLoc)
      return true;
    LastTypeLoc = TypeLoc;
  }
  return false;
}

static bool declaresMultipleFieldsInMacro(const RecordDecl *Decl,
                                          const SourceManager &SrcMgr) {
  SourceLocation LastMacroLoc;
  for (const auto &Field : Decl->fields()) {
    if (!Field->getLocation().isMacroID())
      continue;
    SourceLocation MacroLoc = SrcMgr.getExpansionLoc(Field->getLocation());
    if (LastMacroLoc.isValid() && MacroLoc == LastMacroLoc)
      return true;
    LastMacroLoc = MacroLoc;
  }
  return false;
}

static bool containsPreprocessorDirectives(const RecordDecl *Decl,
                                           const SourceManager &SrcMgr,
                                           const LangOptions &LangOpts) {
  std::pair<FileID, unsigned> FileAndOffset =
      SrcMgr.getDecomposedLoc(Decl->field_begin()->getBeginLoc());
  assert(!Decl->field_empty());
  auto LastField = Decl->field_begin();
  while (std::next(LastField) != Decl->field_end())
    ++LastField;
  unsigned EndOffset = SrcMgr.getFileOffset(LastField->getEndLoc());
  StringRef SrcBuffer = SrcMgr.getBufferData(FileAndOffset.first);
  Lexer L(SrcMgr.getLocForStartOfFile(FileAndOffset.first), LangOpts,
          SrcBuffer.data(), SrcBuffer.data() + FileAndOffset.second,
          SrcBuffer.data() + SrcBuffer.size());
  IdentifierTable Identifiers(LangOpts);
  clang::Token T;
  while (!L.LexFromRawLexer(T) && L.getCurrentBufferOffset() < EndOffset) {
    if (T.getKind() == tok::hash) {
      L.LexFromRawLexer(T);
      if (T.getKind() == tok::raw_identifier) {
        clang::IdentifierInfo &II = Identifiers.get(T.getRawIdentifier());
        if (II.getPPKeywordID() != clang::tok::pp_not_keyword)
          return true;
      }
    }
  }
  return false;
}

static bool isSafeToRewrite(const RecordDecl *Decl, const ASTContext &Context) {
  // All following checks expect at least one field declaration.
  if (Decl->field_empty())
    return true;

  // Don't attempt to rewrite if there is a declaration like 'int a, b;'.
  if (declaresMultipleFieldsInStatement(Decl))
    return false;

  const SourceManager &SrcMgr = Context.getSourceManager();

  // Don't attempt to rewrite if a single macro expansion creates multiple
  // fields.
  if (declaresMultipleFieldsInMacro(Decl, SrcMgr))
    return false;

  // Prevent rewriting if there are preprocessor directives present between the
  // start of the first field and the end of last field.
  if (containsPreprocessorDirectives(Decl, SrcMgr, Context.getLangOpts()))
    return false;

  return true;
}

/// Calculates the new order of fields.
///
/// \returns empty vector if the list of fields doesn't match the definition.
static SmallVector<unsigned, 4>
getNewFieldsOrder(const RecordDecl *Definition,
                  ArrayRef<std::string> DesiredFieldsOrder) {
  assert(Definition && "Definition is null");

  llvm::StringMap<unsigned> NameToIndex;
  for (const auto *Field : Definition->fields())
    NameToIndex[Field->getName()] = Field->getFieldIndex();

  if (DesiredFieldsOrder.size() != NameToIndex.size()) {
    llvm::errs() << "Number of provided fields (" << DesiredFieldsOrder.size()
                 << ") doesn't match definition (" << NameToIndex.size()
                 << ").\n";
    return {};
  }
  SmallVector<unsigned, 4> NewFieldsOrder;
  for (const auto &Name : DesiredFieldsOrder) {
    auto It = NameToIndex.find(Name);
    if (It == NameToIndex.end()) {
      llvm::errs() << "Field " << Name << " not found in definition.\n";
      return {};
    }
    NewFieldsOrder.push_back(It->second);
  }
  assert(NewFieldsOrder.size() == NameToIndex.size());
  return NewFieldsOrder;
}

struct ReorderedStruct {
public:
  ReorderedStruct(const RecordDecl *Decl, ArrayRef<unsigned> NewFieldsOrder)
      : Definition(Decl), NewFieldsOrder(NewFieldsOrder),
        NewFieldsPositions(NewFieldsOrder.size()) {
    for (unsigned I = 0; I < NewFieldsPositions.size(); ++I)
      NewFieldsPositions[NewFieldsOrder[I]] = I;
  }

  /// Compares compatible designators according to the new struct order.
  /// Returns a negative value if Lhs < Rhs, positive value if Lhs > Rhs and 0
  /// if they are equal.
  bool operator()(const Designator &Lhs, const Designator &Rhs) const;

  /// Compares compatible designator lists according to the new struct order.
  /// Returns a negative value if Lhs < Rhs, positive value if Lhs > Rhs and 0
  /// if they are equal.
  bool operator()(const Designators &Lhs, const Designators &Rhs) const;

  const RecordDecl *Definition;
  ArrayRef<unsigned> NewFieldsOrder;
  SmallVector<unsigned, 4> NewFieldsPositions;
};

bool ReorderedStruct::operator()(const Designator &Lhs,
                                 const Designator &Rhs) const {
  switch (Lhs.getTag()) {
  case Designator::STRUCT:
    assert(Rhs.getTag() == Designator::STRUCT && "Incompatible designators");
    assert(Lhs.getStructDecl() == Rhs.getStructDecl() &&
           "Incompatible structs");
    // Use the new layout for reordered struct.
    if (Definition == Lhs.getStructDecl()) {
      return NewFieldsPositions[Lhs.getStructIter()->getFieldIndex()] <
             NewFieldsPositions[Rhs.getStructIter()->getFieldIndex()];
    }
    return Lhs.getStructIter()->getFieldIndex() <
           Rhs.getStructIter()->getFieldIndex();
  case Designator::ARRAY:
  case Designator::ARRAY_RANGE:
    // Array designators can be compared to array range designators.
    assert((Rhs.getTag() == Designator::ARRAY ||
            Rhs.getTag() == Designator::ARRAY_RANGE) &&
           "Incompatible designators");
    size_t LhsIdx = Lhs.getTag() == Designator::ARRAY
                        ? Lhs.getArrayIndex()
                        : Lhs.getArrayRangeStart();
    size_t RhsIdx = Rhs.getTag() == Designator::ARRAY
                        ? Rhs.getArrayIndex()
                        : Rhs.getArrayRangeStart();
    return LhsIdx < RhsIdx;
  }
  llvm_unreachable("Invalid designator tag");
}

bool ReorderedStruct::operator()(const Designators &Lhs,
                                 const Designators &Rhs) const {
  return std::lexicographical_compare(Lhs.begin(), Lhs.end(), Rhs.begin(),
                                      Rhs.end(), *this);
}

// FIXME: error-handling
/// Replaces a range of source code by the specified text.
static void
addReplacement(SourceRange Old, StringRef New, const ASTContext &Context,
               std::map<std::string, tooling::Replacements> &Replacements) {
  tooling::Replacement R(Context.getSourceManager(),
                         CharSourceRange::getTokenRange(Old), New,
                         Context.getLangOpts());
  consumeError(Replacements[std::string(R.getFilePath())].add(R));
}

/// Replaces one range of source code by another and adds a prefix.
static void
addReplacement(SourceRange Old, SourceRange New, StringRef Prefix,
               const ASTContext &Context,
               std::map<std::string, tooling::Replacements> &Replacements) {
  std::string NewText =
      (Prefix + Lexer::getSourceText(CharSourceRange::getTokenRange(New),
                                     Context.getSourceManager(),
                                     Context.getLangOpts()))
          .str();
  addReplacement(Old, NewText, Context, Replacements);
}

/// Replaces one range of source code by another.
static void
addReplacement(SourceRange Old, SourceRange New, const ASTContext &Context,
               std::map<std::string, tooling::Replacements> &Replacements) {
  if (Old.getBegin().isMacroID())
    Old = Context.getSourceManager().getExpansionRange(Old).getAsRange();
  if (New.getBegin().isMacroID())
    New = Context.getSourceManager().getExpansionRange(New).getAsRange();
  StringRef NewText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(New),
                           Context.getSourceManager(), Context.getLangOpts());
  addReplacement(Old, NewText.str(), Context, Replacements);
}

/// Find all member fields used in the given init-list initializer expr
/// that belong to the same record
///
/// \returns a set of field declarations, empty if none were present
static SmallSetVector<FieldDecl *, 1>
findMembersUsedInInitExpr(const CXXCtorInitializer *Initializer,
                          ASTContext &Context) {
  SmallSetVector<FieldDecl *, 1> Results;
  // Note that this does not pick up member fields of base classes since
  // for those accesses Sema::PerformObjectMemberConversion always inserts an
  // UncheckedDerivedToBase ImplicitCastExpr between the this expr and the
  // object expression
  auto FoundExprs = match(
      traverse(
          TK_AsIs,
          findAll(memberExpr(hasObjectExpression(cxxThisExpr())).bind("ME"))),
      *Initializer->getInit(), Context);
  for (BoundNodes &BN : FoundExprs)
    if (auto *MemExpr = BN.getNodeAs<MemberExpr>("ME"))
      if (auto *FD = dyn_cast<FieldDecl>(MemExpr->getMemberDecl()))
        Results.insert(FD);
  return Results;
}

/// Returns the start of the leading comments before `Loc`.
static SourceLocation getStartOfLeadingComment(SourceLocation Loc,
                                               const SourceManager &SM,
                                               const LangOptions &LangOpts) {
  // We consider any leading comment token that is on the same line or
  // indented similarly to the first comment to be part of the leading comment.
  const unsigned Line = SM.getPresumedLineNumber(Loc);
  const unsigned Column = SM.getPresumedColumnNumber(Loc);
  std::optional<Token> Tok =
      Lexer::findPreviousToken(Loc, SM, LangOpts, /*IncludeComments=*/true);
  while (Tok && Tok->is(tok::comment)) {
    const SourceLocation CommentLoc =
        Lexer::GetBeginningOfToken(Tok->getLocation(), SM, LangOpts);
    if (SM.getPresumedLineNumber(CommentLoc) != Line &&
        SM.getPresumedColumnNumber(CommentLoc) != Column) {
      break;
    }
    Loc = CommentLoc;
    Tok = Lexer::findPreviousToken(Loc, SM, LangOpts, /*IncludeComments=*/true);
  }
  return Loc;
}

/// Returns the end of the trailing comments after `Loc`.
static SourceLocation getEndOfTrailingComment(SourceLocation Loc,
                                              const SourceManager &SM,
                                              const LangOptions &LangOpts) {
  // We consider any following comment token that is indented more than the
  // first comment to be part of the trailing comment.
  const unsigned Column = SM.getPresumedColumnNumber(Loc);
  std::optional<Token> Tok =
      Lexer::findNextToken(Loc, SM, LangOpts, /*IncludeComments=*/true);
  while (Tok && Tok->is(tok::comment) &&
         SM.getPresumedColumnNumber(Tok->getLocation()) > Column) {
    Loc = Tok->getEndLoc();
    Tok = Lexer::findNextToken(Loc, SM, LangOpts, /*IncludeComments=*/true);
  }
  return Loc;
}

/// Returns the full source range for the field declaration up to (including)
/// the trailing semicolumn, including potential macro invocations,
/// e.g. `int a GUARDED_BY(mu);`. If there is a trailing comment, include it.
static SourceRange getFullFieldSourceRange(const FieldDecl &Field,
                                           const ASTContext &Context) {
  const SourceRange Range = Field.getSourceRange();
  SourceLocation Begin = Range.getBegin();
  SourceLocation End = Range.getEnd();
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  while (true) {
    std::optional<Token> CurrentToken = Lexer::findNextToken(End, SM, LangOpts);

    if (!CurrentToken)
      return SourceRange(Begin, End);

    if (CurrentToken->is(tok::eof))
      return Range; // Something is wrong, return the original range.

    End = CurrentToken->getLastLoc();

    if (CurrentToken->is(tok::semi))
      break;
  }
  Begin = getStartOfLeadingComment(Begin, SM, LangOpts);
  End = getEndOfTrailingComment(End, SM, LangOpts);
  return SourceRange(Begin, End);
}

/// Reorders fields in the definition of a struct/class.
///
/// At the moment reordering of fields with
/// different accesses (public/protected/private) is not supported.
/// \returns true on success.
static bool reorderFieldsInDefinition(
    const ReorderedStruct &RS, const ASTContext &Context,
    std::map<std::string, tooling::Replacements> &Replacements) {
  assert(RS.Definition && "Definition is null");

  SmallVector<const FieldDecl *, 10> Fields;
  for (const auto *Field : RS.Definition->fields())
    Fields.push_back(Field);

  // Check that the permutation of the fields doesn't change the accesses
  for (const auto *Field : RS.Definition->fields()) {
    const auto FieldIndex = Field->getFieldIndex();
    if (Field->getAccess() !=
        Fields[RS.NewFieldsOrder[FieldIndex]]->getAccess()) {
      llvm::errs() << "Currently reordering of fields with different accesses "
                      "is not supported\n";
      return false;
    }
  }

  for (const auto *Field : RS.Definition->fields()) {
    const auto FieldIndex = Field->getFieldIndex();
    if (FieldIndex == RS.NewFieldsOrder[FieldIndex])
      continue;
    addReplacement(getFullFieldSourceRange(*Field, Context),
                   getFullFieldSourceRange(
                       *Fields[RS.NewFieldsOrder[FieldIndex]], Context),
                   Context, Replacements);
  }
  return true;
}

/// Reorders initializers in a C++ struct/class constructor.
///
/// A constructor can have initializers for an arbitrary subset of the class's
/// fields. Thus, we need to ensure that we reorder just the initializers that
/// are present.
static void reorderFieldsInConstructor(
    const CXXConstructorDecl *CtorDecl, const ReorderedStruct &RS,
    ASTContext &Context,
    std::map<std::string, tooling::Replacements> &Replacements) {
  assert(CtorDecl && "Constructor declaration is null");
  if (CtorDecl->isImplicit() || CtorDecl->getNumCtorInitializers() <= 1)
    return;

  // The method FunctionDecl::isThisDeclarationADefinition returns false
  // for a defaulted function unless that function has been implicitly defined.
  // Thus this assert needs to be after the previous checks.
  assert(CtorDecl->isThisDeclarationADefinition() && "Not a definition");

  SmallVector<const CXXCtorInitializer *, 10> OldWrittenInitializersOrder;
  SmallVector<const CXXCtorInitializer *, 10> NewWrittenInitializersOrder;
  for (const auto *Initializer : CtorDecl->inits()) {
    if (!Initializer->isMemberInitializer() || !Initializer->isWritten())
      continue;

    // Warn if this reordering violates initialization expr dependencies.
    const FieldDecl *ThisM = Initializer->getMember();
    const auto UsedMembers = findMembersUsedInInitExpr(Initializer, Context);
    for (const FieldDecl *UM : UsedMembers) {
      if (RS.NewFieldsPositions[UM->getFieldIndex()] >
          RS.NewFieldsPositions[ThisM->getFieldIndex()]) {
        DiagnosticsEngine &DiagEngine = Context.getDiagnostics();
        auto Description = ("reordering field " + UM->getName() + " after " +
                            ThisM->getName() + " makes " + UM->getName() +
                            " uninitialized when used in init expression")
                               .str();
        unsigned ID = DiagEngine.getDiagnosticIDs()->getCustomDiagID(
            DiagnosticIDs::Warning, Description);
        DiagEngine.Report(Initializer->getSourceLocation(), ID);
      }
    }

    OldWrittenInitializersOrder.push_back(Initializer);
    NewWrittenInitializersOrder.push_back(Initializer);
  }
  auto ByFieldNewPosition = [&](const CXXCtorInitializer *LHS,
                                const CXXCtorInitializer *RHS) {
    assert(LHS && RHS);
    return RS.NewFieldsPositions[LHS->getMember()->getFieldIndex()] <
           RS.NewFieldsPositions[RHS->getMember()->getFieldIndex()];
  };
  llvm::sort(NewWrittenInitializersOrder, ByFieldNewPosition);
  assert(OldWrittenInitializersOrder.size() ==
         NewWrittenInitializersOrder.size());
  for (unsigned i = 0, e = NewWrittenInitializersOrder.size(); i < e; ++i)
    if (OldWrittenInitializersOrder[i] != NewWrittenInitializersOrder[i])
      addReplacement(OldWrittenInitializersOrder[i]->getSourceRange(),
                     NewWrittenInitializersOrder[i]->getSourceRange(), Context,
                     Replacements);
}

/// Replacement for broken InitListExpr::isExplicit function.
/// FIXME: Remove when InitListExpr::isExplicit is fixed.
static bool isImplicitILE(const InitListExpr *ILE, const ASTContext &Context) {
  // The ILE is implicit if either:
  // - The left brace loc of the ILE matches the start of first init expression
  //   (for non designated decls)
  // - The right brace loc of the ILE matches the end of first init expression
  //   (for designated decls)
  // The first init expression should be taken from the syntactic form, but
  // since the ILE could be implicit, there might not be a syntactic form.
  // For that reason we have to check against all init expressions.
  for (const Expr *Init : ILE->inits()) {
    if (ILE->getLBraceLoc() == Init->getBeginLoc() ||
        ILE->getRBraceLoc() == Init->getEndLoc())
      return true;
  }
  return false;
}

/// Finds the semantic form of the first explicit ancestor of the given
/// initializer list including itself.
static const InitListExpr *getExplicitILE(const InitListExpr *ILE,
                                          ASTContext &Context) {
  if (!isImplicitILE(ILE, Context))
    return ILE;
  const InitListExpr *TopLevelILE = ILE;
  DynTypedNodeList Parents = Context.getParents(*TopLevelILE);
  while (!Parents.empty() && Parents.begin()->get<InitListExpr>()) {
    TopLevelILE = Parents.begin()->get<InitListExpr>();
    Parents = Context.getParents(*TopLevelILE);
    if (!isImplicitILE(TopLevelILE, Context))
      break;
  }
  if (!TopLevelILE->isSemanticForm()) {
    return TopLevelILE->getSemanticForm();
  }
  return TopLevelILE;
}

static void reportError(const Twine &Message, SourceLocation Loc,
                        const SourceManager &SM) {
  if (Loc.isValid()) {
    llvm::errs() << SM.getFilename(Loc) << ":" << SM.getPresumedLineNumber(Loc)
                 << ":" << SM.getPresumedColumnNumber(Loc) << ": ";
  }
  llvm::errs() << Message;
}

/// Reorders initializers in the brace initialization of an aggregate.
///
/// At the moment partial initialization is not supported.
/// \returns true on success
static bool reorderFieldsInInitListExpr(
    const InitListExpr *InitListEx, const ReorderedStruct &RS,
    ASTContext &Context,
    std::map<std::string, tooling::Replacements> &Replacements) {
  assert(InitListEx && "Init list expression is null");
  // Only process semantic forms of initializer lists.
  if (!InitListEx->isSemanticForm()) {
    return true;
  }

  // If there are no initializers we do not need to change anything.
  if (!InitListEx->getNumInits())
    return true;

  // We care only about InitListExprs which originate from source code.
  // Implicit InitListExprs are created by the semantic analyzer.
  // We find the first parent InitListExpr that exists in source code and
  // process it. This is necessary because of designated initializer lists and
  // possible omitted braces.
  InitListEx = getExplicitILE(InitListEx, Context);

  // Find if there are any designated initializations or implicit values. If all
  // initializers are present and none have designators then just reorder them
  // normally. Otherwise, designators are added to all initializers and they are
  // sorted in the new order.
  bool HasImplicitInit = false;
  bool HasDesignatedInit = false;
  // The method InitListExpr::getSyntacticForm may return nullptr indicating
  // that the current initializer list also serves as its syntactic form.
  const InitListExpr *SyntacticInitListEx = InitListEx;
  if (const InitListExpr *SynILE = InitListEx->getSyntacticForm()) {
    // Do not rewrite zero initializers. This check is only valid for syntactic
    // forms.
    if (SynILE->isIdiomaticZeroInitializer(Context.getLangOpts()))
      return true;

    HasImplicitInit = InitListEx->getNumInits() != SynILE->getNumInits();
    HasDesignatedInit = llvm::any_of(SynILE->inits(), [](const Expr *Init) {
      return isa<DesignatedInitExpr>(Init);
    });

    SyntacticInitListEx = SynILE;
  } else {
    // If there is no syntactic form, there can be no designators. Instead,
    // there might be implicit values.
    HasImplicitInit =
        (RS.NewFieldsOrder.size() != InitListEx->getNumInits()) ||
        llvm::any_of(InitListEx->inits(), [&Context](const Expr *Init) {
          return isa<ImplicitValueInitExpr>(Init) ||
                 (isa<InitListExpr>(Init) &&
                  isImplicitILE(dyn_cast<InitListExpr>(Init), Context));
        });
  }

  if (HasImplicitInit || HasDesignatedInit) {
    // Designators are only supported from C++20.
    if (!HasDesignatedInit && Context.getLangOpts().CPlusPlus &&
        !Context.getLangOpts().CPlusPlus20) {
      reportError(
          "Only full initialization without implicit values is supported\n",
          InitListEx->getBeginLoc(), Context.getSourceManager());
      return false;
    }

    // Handle case when some fields are designated. Some fields can be
    // missing. Insert any missing designators and reorder the expressions
    // according to the new order.
    std::optional<Designators> CurrentDesignators;
    // Remember each initializer expression along with its designators. They are
    // sorted later to determine the correct order.
    std::vector<std::pair<Designators, const Expr *>> Rewrites;
    for (const Expr *Init : SyntacticInitListEx->inits()) {
      if (const auto *DIE = dyn_cast_or_null<DesignatedInitExpr>(Init)) {
        CurrentDesignators.emplace(DIE, SyntacticInitListEx, &Context);
        if (!CurrentDesignators->isValid()) {
          reportError("Unsupported initializer list\n", DIE->getBeginLoc(),
                      Context.getSourceManager());
          return false;
        }

        // Use the child of the DesignatedInitExpr. This way designators are
        // always replaced.
        Rewrites.emplace_back(*CurrentDesignators, DIE->getInit());
      } else {
        // If designators are not initialized then initialize to the first
        // field, otherwise move the next field.
        if (!CurrentDesignators) {
          CurrentDesignators.emplace(Init, SyntacticInitListEx, &Context);
          if (!CurrentDesignators->isValid()) {
            reportError("Unsupported initializer list\n",
                        InitListEx->getBeginLoc(), Context.getSourceManager());
            return false;
          }
        } else if (!CurrentDesignators->advanceToNextField(Init)) {
          reportError("Unsupported initializer list\n",
                      InitListEx->getBeginLoc(), Context.getSourceManager());
          return false;
        }

        // Do not rewrite implicit values. They just had to be processed to
        // find the correct designator.
        if (!isa<ImplicitValueInitExpr>(Init))
          Rewrites.emplace_back(*CurrentDesignators, Init);
      }
    }

    // Sort the designators according to the new order.
    llvm::stable_sort(Rewrites, [&RS](const auto &Lhs, const auto &Rhs) {
      return RS(Lhs.first, Rhs.first);
    });

    for (unsigned i = 0, e = Rewrites.size(); i < e; ++i) {
      addReplacement(SyntacticInitListEx->getInit(i)->getSourceRange(),
                     Rewrites[i].second->getSourceRange(),
                     Rewrites[i].first.toString(), Context, Replacements);
    }
  } else {
    // Handle excess initializers by leaving them unchanged.
    assert(SyntacticInitListEx->getNumInits() >= InitListEx->getNumInits());

    // All field initializers are present and none have designators. They can be
    // reordered normally.
    for (unsigned i = 0, e = RS.NewFieldsOrder.size(); i < e; ++i) {
      if (i != RS.NewFieldsOrder[i])
        addReplacement(SyntacticInitListEx->getInit(i)->getSourceRange(),
                       SyntacticInitListEx->getInit(RS.NewFieldsOrder[i])
                           ->getSourceRange(),
                       Context, Replacements);
    }
  }
  return true;
}

namespace {
class ReorderingConsumer : public ASTConsumer {
  StringRef RecordName;
  ArrayRef<std::string> DesiredFieldsOrder;
  std::map<std::string, tooling::Replacements> &Replacements;

public:
  ReorderingConsumer(StringRef RecordName,
                     ArrayRef<std::string> DesiredFieldsOrder,
                     std::map<std::string, tooling::Replacements> &Replacements)
      : RecordName(RecordName), DesiredFieldsOrder(DesiredFieldsOrder),
        Replacements(Replacements) {}

  ReorderingConsumer(const ReorderingConsumer &) = delete;
  ReorderingConsumer &operator=(const ReorderingConsumer &) = delete;

  void HandleTranslationUnit(ASTContext &Context) override {
    const RecordDecl *RD = findDefinition(RecordName, Context);
    if (!RD)
      return;
    if (!isSafeToRewrite(RD, Context))
      return;
    SmallVector<unsigned, 4> NewFieldsOrder =
        getNewFieldsOrder(RD, DesiredFieldsOrder);
    if (NewFieldsOrder.empty())
      return;
    ReorderedStruct RS{RD, NewFieldsOrder};

    if (!reorderFieldsInDefinition(RS, Context, Replacements))
      return;

    // CXXRD will be nullptr if C code (not C++) is being processed.
    const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD);
    if (CXXRD)
      for (const auto *C : CXXRD->ctors())
        if (const auto *D = dyn_cast<CXXConstructorDecl>(C->getDefinition()))
          reorderFieldsInConstructor(cast<const CXXConstructorDecl>(D), RS,
                                     Context, Replacements);

    // We only need to reorder init list expressions for
    // plain C structs or C++ aggregate types.
    // For other types the order of constructor parameters is used,
    // which we don't change at the moment.
    // Now (v0) partial initialization is not supported.
    if (!CXXRD || CXXRD->isAggregate()) {
      for (auto Result :
           match(initListExpr(hasType(equalsNode(RD))).bind("initListExpr"),
                 Context))
        if (!reorderFieldsInInitListExpr(
                Result.getNodeAs<InitListExpr>("initListExpr"), RS, Context,
                Replacements)) {
          Replacements.clear();
          return;
        }
    }
  }
};
} // end anonymous namespace

std::unique_ptr<ASTConsumer> ReorderFieldsAction::newASTConsumer() {
  return std::make_unique<ReorderingConsumer>(RecordName, DesiredFieldsOrder,
                                               Replacements);
}

} // namespace reorder_fields
} // namespace clang
