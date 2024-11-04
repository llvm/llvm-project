//===--- PreferMemberInitializerCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferMemberInitializerCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseMap.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

static bool isControlStatement(const Stmt *S) {
  return isa<IfStmt, SwitchStmt, ForStmt, WhileStmt, DoStmt, ReturnStmt,
             GotoStmt, CXXTryStmt, CXXThrowExpr>(S);
}

static bool isNoReturnCallStatement(const Stmt *S) {
  const auto *Call = dyn_cast<CallExpr>(S);
  if (!Call)
    return false;

  const FunctionDecl *Func = Call->getDirectCallee();
  if (!Func)
    return false;

  return Func->isNoReturn();
}

namespace {

AST_MATCHER_P(FieldDecl, indexNotLessThan, unsigned, Index) {
  return Node.getFieldIndex() >= Index;
}

enum class AssignedLevel {
  // Field is not assigned.
  None,
  // Field is assigned.
  Default,
  // Assignment of field has side effect:
  // - assign to reference.
  // FIXME: support other side effect.
  HasSideEffect,
  // Assignment of field has data dependence.
  HasDependence,
};

} // namespace

static bool canAdvanceAssignment(AssignedLevel Level) {
  return Level == AssignedLevel::None || Level == AssignedLevel::Default;
}

// Checks if Field is initialised using a field that will be initialised after
// it.
// TODO: Probably should guard against function calls that could have side
// effects or if they do reference another field that's initialized before
// this field, but is modified before the assignment.
static void updateAssignmentLevel(
    const FieldDecl *Field, const Expr *Init, const CXXConstructorDecl *Ctor,
    llvm::DenseMap<const FieldDecl *, AssignedLevel> &AssignedFields) {
  auto It = AssignedFields.find(Field);
  if (It == AssignedFields.end())
    It = AssignedFields.insert({Field, AssignedLevel::None}).first;

  if (!canAdvanceAssignment(It->second))
    // fast path for already decided field.
    return;

  if (Field->getType().getCanonicalType()->isReferenceType()) {
    // assign to reference type twice cannot be simplified to once.
    It->second = AssignedLevel::HasSideEffect;
    return;
  }

  auto MemberMatcher =
      memberExpr(hasObjectExpression(cxxThisExpr()),
                 member(fieldDecl(indexNotLessThan(Field->getFieldIndex()))));
  auto DeclMatcher = declRefExpr(
      to(varDecl(unless(parmVarDecl()), hasDeclContext(equalsNode(Ctor)))));
  const bool HasDependence = !match(expr(anyOf(MemberMatcher, DeclMatcher,
                                               hasDescendant(MemberMatcher),
                                               hasDescendant(DeclMatcher))),
                                    *Init, Field->getASTContext())
                                  .empty();
  if (HasDependence) {
    It->second = AssignedLevel::HasDependence;
    return;
  }
}

struct AssignmentPair {
  const FieldDecl *Field;
  const Expr *Init;
};

static std::optional<AssignmentPair>
isAssignmentToMemberOf(const CXXRecordDecl *Rec, const Stmt *S,
                       const CXXConstructorDecl *Ctor) {
  if (const auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() != BO_Assign)
      return {};

    const auto *ME = dyn_cast<MemberExpr>(BO->getLHS()->IgnoreParenImpCasts());
    if (!ME)
      return {};

    const auto *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
    if (!Field)
      return {};

    if (!isa<CXXThisExpr>(ME->getBase()))
      return {};
    const Expr *Init = BO->getRHS()->IgnoreParenImpCasts();
    return AssignmentPair{Field, Init};
  }
  if (const auto *COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (COCE->getOperator() != OO_Equal)
      return {};

    const auto *ME =
        dyn_cast<MemberExpr>(COCE->getArg(0)->IgnoreParenImpCasts());
    if (!ME)
      return {};

    const auto *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
    if (!Field)
      return {};

    if (!isa<CXXThisExpr>(ME->getBase()))
      return {};
    const Expr *Init = COCE->getArg(1)->IgnoreParenImpCasts();
    return AssignmentPair{Field, Init};
  }
  return {};
}

PreferMemberInitializerCheck::PreferMemberInitializerCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void PreferMemberInitializerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(cxxConstructorDecl(hasBody(compoundStmt()),
                                        unless(isInstantiated()),
                                        unless(isDelegatingConstructor()))
                         .bind("ctor"),
                     this);
}

void PreferMemberInitializerCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  const auto *Body = cast<CompoundStmt>(Ctor->getBody());

  const CXXRecordDecl *Class = Ctor->getParent();
  bool FirstToCtorInits = true;

  llvm::DenseMap<const FieldDecl *, AssignedLevel> AssignedFields{};

  for (const CXXCtorInitializer *Init : Ctor->inits())
    if (FieldDecl *Field = Init->getMember())
      updateAssignmentLevel(Field, Init->getInit(), Ctor, AssignedFields);

  for (const Stmt *S : Body->body()) {
    if (S->getBeginLoc().isMacroID()) {
      StringRef MacroName = Lexer::getImmediateMacroName(
          S->getBeginLoc(), *Result.SourceManager, getLangOpts());
      if (MacroName.contains_insensitive("assert"))
        return;
    }
    if (isControlStatement(S))
      return;

    if (isNoReturnCallStatement(S))
      return;

    if (const auto *CondOp = dyn_cast<ConditionalOperator>(S)) {
      if (isNoReturnCallStatement(CondOp->getLHS()) ||
          isNoReturnCallStatement(CondOp->getRHS()))
        return;
    }

    std::optional<AssignmentPair> AssignmentToMember =
        isAssignmentToMemberOf(Class, S, Ctor);
    if (!AssignmentToMember)
      continue;
    const FieldDecl *Field = AssignmentToMember->Field;
    const Expr *InitValue = AssignmentToMember->Init;
    updateAssignmentLevel(Field, InitValue, Ctor, AssignedFields);
    if (!canAdvanceAssignment(AssignedFields[Field]))
      continue;

    StringRef InsertPrefix = "";
    bool HasInitAlready = false;
    SourceLocation InsertPos;
    SourceRange ReplaceRange;
    bool AddComma = false;
    bool AddBrace = false;
    bool InvalidFix = false;
    unsigned Index = Field->getFieldIndex();
    const CXXCtorInitializer *LastInListInit = nullptr;
    for (const CXXCtorInitializer *Init : Ctor->inits()) {
      if (!Init->isWritten() || Init->isInClassMemberInitializer())
        continue;
      if (Init->getMember() == Field) {
        HasInitAlready = true;
        if (isa<ImplicitValueInitExpr>(Init->getInit()))
          InsertPos = Init->getRParenLoc();
        else {
          ReplaceRange = Init->getInit()->getSourceRange();
          AddBrace = isa<InitListExpr>(Init->getInit());
        }
        break;
      }
      if (Init->isMemberInitializer() &&
          Index < Init->getMember()->getFieldIndex()) {
        InsertPos = Init->getSourceLocation();
        // There are initializers after the one we are inserting, so add a
        // comma after this insertion in order to not break anything.
        AddComma = true;
        break;
      }
      LastInListInit = Init;
    }
    if (HasInitAlready) {
      if (InsertPos.isValid())
        InvalidFix |= InsertPos.isMacroID();
      else
        InvalidFix |= ReplaceRange.getBegin().isMacroID() ||
                      ReplaceRange.getEnd().isMacroID();
    } else {
      if (InsertPos.isInvalid()) {
        if (LastInListInit) {
          InsertPos =
              Lexer::getLocForEndOfToken(LastInListInit->getRParenLoc(), 0,
                                         *Result.SourceManager, getLangOpts());
          // Inserting after the last constructor initializer, so we need a
          // comma.
          InsertPrefix = ", ";
        } else {
          InsertPos = Lexer::getLocForEndOfToken(
              Ctor->getTypeSourceInfo()
                  ->getTypeLoc()
                  .getAs<clang::FunctionTypeLoc>()
                  .getLocalRangeEnd(),
              0, *Result.SourceManager, getLangOpts());

          // If this is first time in the loop, there are no initializers so
          // `:` declares member initialization list. If this is a
          // subsequent pass then we have already inserted a `:` so continue
          // with a comma.
          InsertPrefix = FirstToCtorInits ? " : " : ", ";
        }
      }
      InvalidFix |= InsertPos.isMacroID();
    }

    SourceLocation SemiColonEnd;
    if (auto NextToken = Lexer::findNextToken(
            S->getEndLoc(), *Result.SourceManager, getLangOpts()))
      SemiColonEnd = NextToken->getEndLoc();
    else
      InvalidFix = true;

    auto Diag = diag(S->getBeginLoc(), "%0 should be initialized in a member"
                                       " initializer of the constructor")
                << Field;
    if (InvalidFix)
      continue;
    StringRef NewInit = Lexer::getSourceText(
        Result.SourceManager->getExpansionRange(InitValue->getSourceRange()),
        *Result.SourceManager, getLangOpts());
    if (HasInitAlready) {
      if (InsertPos.isValid())
        Diag << FixItHint::CreateInsertion(InsertPos, NewInit);
      else if (AddBrace)
        Diag << FixItHint::CreateReplacement(ReplaceRange,
                                             ("{" + NewInit + "}").str());
      else
        Diag << FixItHint::CreateReplacement(ReplaceRange, NewInit);
    } else {
      SmallString<128> Insertion({InsertPrefix, Field->getName(), "(", NewInit,
                                  AddComma ? "), " : ")"});
      Diag << FixItHint::CreateInsertion(InsertPos, Insertion,
                                         FirstToCtorInits);
      FirstToCtorInits = areDiagsSelfContained();
    }
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(S->getBeginLoc(), SemiColonEnd));
  }
}

} // namespace clang::tidy::cppcoreguidelines
