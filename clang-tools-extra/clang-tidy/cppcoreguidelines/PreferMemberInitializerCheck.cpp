//===--- PreferMemberInitializerCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferMemberInitializerCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

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

static bool isLiteral(const Expr *E) {
  return isa<StringLiteral, CharacterLiteral, IntegerLiteral, FloatingLiteral,
             CXXBoolLiteralExpr, CXXNullPtrLiteralExpr>(E);
}

static bool isUnaryExprOfLiteral(const Expr *E) {
  if (const auto *UnOp = dyn_cast<UnaryOperator>(E))
    return isLiteral(UnOp->getSubExpr());
  return false;
}

static bool shouldBeDefaultMemberInitializer(const Expr *Value) {
  if (isLiteral(Value) || isUnaryExprOfLiteral(Value))
    return true;

  if (const auto *DRE = dyn_cast<DeclRefExpr>(Value))
    return isa<EnumConstantDecl>(DRE->getDecl());

  return false;
}

namespace {
AST_MATCHER_P(FieldDecl, indexNotLessThan, unsigned, Index) {
  return Node.getFieldIndex() >= Index;
}
} // namespace

// Checks if Field is initialised using a field that will be initialised after
// it.
// TODO: Probably should guard against function calls that could have side
// effects or if they do reference another field that's initialized before this
// field, but is modified before the assignment.
static bool isSafeAssignment(const FieldDecl *Field, const Expr *Init,
                             const CXXConstructorDecl *Context) {

  auto MemberMatcher =
      memberExpr(hasObjectExpression(cxxThisExpr()),
                 member(fieldDecl(indexNotLessThan(Field->getFieldIndex()))));

  auto DeclMatcher = declRefExpr(
      to(varDecl(unless(parmVarDecl()), hasDeclContext(equalsNode(Context)))));

  return match(expr(anyOf(MemberMatcher, DeclMatcher,
                          hasDescendant(MemberMatcher),
                          hasDescendant(DeclMatcher))),
               *Init, Field->getASTContext())
      .empty();
}

static std::pair<const FieldDecl *, const Expr *>
isAssignmentToMemberOf(const CXXRecordDecl *Rec, const Stmt *S,
                       const CXXConstructorDecl *Ctor) {
  if (const auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() != BO_Assign)
      return std::make_pair(nullptr, nullptr);

    const auto *ME = dyn_cast<MemberExpr>(BO->getLHS()->IgnoreParenImpCasts());
    if (!ME)
      return std::make_pair(nullptr, nullptr);

    const auto *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
    if (!Field)
      return std::make_pair(nullptr, nullptr);

    if (!isa<CXXThisExpr>(ME->getBase()))
      return std::make_pair(nullptr, nullptr);
    const Expr *Init = BO->getRHS()->IgnoreParenImpCasts();
    if (isSafeAssignment(Field, Init, Ctor))
      return std::make_pair(Field, Init);
  } else if (const auto *COCE = dyn_cast<CXXOperatorCallExpr>(S)) {
    if (COCE->getOperator() != OO_Equal)
      return std::make_pair(nullptr, nullptr);

    const auto *ME =
        dyn_cast<MemberExpr>(COCE->getArg(0)->IgnoreParenImpCasts());
    if (!ME)
      return std::make_pair(nullptr, nullptr);

    const auto *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
    if (!Field)
      return std::make_pair(nullptr, nullptr);

    if (!isa<CXXThisExpr>(ME->getBase()))
      return std::make_pair(nullptr, nullptr);
    const Expr *Init = COCE->getArg(1)->IgnoreParenImpCasts();
    if (isSafeAssignment(Field, Init, Ctor))
      return std::make_pair(Field, Init);
  }

  return std::make_pair(nullptr, nullptr);
}

PreferMemberInitializerCheck::PreferMemberInitializerCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IsUseDefaultMemberInitEnabled(
          Context->isCheckEnabled("modernize-use-default-member-init")),
      UseAssignment(
          Options.get("UseAssignment",
                      OptionsView("modernize-use-default-member-init",
                                  Context->getOptions().CheckOptions, Context)
                          .get("UseAssignment", false))) {}

void PreferMemberInitializerCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UseAssignment", UseAssignment);
}

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

    const FieldDecl *Field = nullptr;
    const Expr *InitValue = nullptr;
    std::tie(Field, InitValue) = isAssignmentToMemberOf(Class, S, Ctor);
    if (Field) {
      if (IsUseDefaultMemberInitEnabled && getLangOpts().CPlusPlus11 &&
          Ctor->isDefaultConstructor() &&
          (getLangOpts().CPlusPlus20 || !Field->isBitField()) &&
          !Field->hasInClassInitializer() &&
          (!isa<RecordDecl>(Class->getDeclContext()) ||
           !cast<RecordDecl>(Class->getDeclContext())->isUnion()) &&
          shouldBeDefaultMemberInitializer(InitValue)) {

        bool InvalidFix = false;
        SourceLocation FieldEnd =
            Lexer::getLocForEndOfToken(Field->getSourceRange().getEnd(), 0,
                                       *Result.SourceManager, getLangOpts());
        InvalidFix |= FieldEnd.isInvalid() || FieldEnd.isMacroID();
        SourceLocation SemiColonEnd;
        if (auto NextToken = Lexer::findNextToken(
                S->getEndLoc(), *Result.SourceManager, getLangOpts()))
          SemiColonEnd = NextToken->getEndLoc();
        else
          InvalidFix = true;
        auto Diag =
            diag(S->getBeginLoc(), "%0 should be initialized in an in-class"
                                   " default member initializer")
            << Field;
        if (InvalidFix)
          continue;
        CharSourceRange StmtRange =
            CharSourceRange::getCharRange(S->getBeginLoc(), SemiColonEnd);

        SmallString<128> Insertion(
            {UseAssignment ? " = " : "{",
             Lexer::getSourceText(
                 CharSourceRange(InitValue->getSourceRange(), true),
                 *Result.SourceManager, getLangOpts()),
             UseAssignment ? "" : "}"});

        Diag << FixItHint::CreateInsertion(FieldEnd, Insertion)
             << FixItHint::CreateRemoval(StmtRange);

      } else {
        StringRef InsertPrefix = "";
        bool HasInitAlready = false;
        SourceLocation InsertPos;
        SourceRange ReplaceRange;
        bool AddComma = false;
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
              InsertPos = Lexer::getLocForEndOfToken(
                  LastInListInit->getRParenLoc(), 0, *Result.SourceManager,
                  getLangOpts());
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

        auto Diag =
            diag(S->getBeginLoc(), "%0 should be initialized in a member"
                                   " initializer of the constructor")
            << Field;
        if (InvalidFix)
          continue;
        StringRef NewInit = Lexer::getSourceText(
            CharSourceRange(InitValue->getSourceRange(), true),
            *Result.SourceManager, getLangOpts());
        if (HasInitAlready) {
          if (InsertPos.isValid())
            Diag << FixItHint::CreateInsertion(InsertPos, NewInit);
          else
            Diag << FixItHint::CreateReplacement(ReplaceRange, NewInit);
        } else {
          SmallString<128> Insertion({InsertPrefix, Field->getName(), "(",
                                      NewInit, AddComma ? "), " : ")"});
          Diag << FixItHint::CreateInsertion(InsertPos, Insertion,
                                             FirstToCtorInits);
          FirstToCtorInits = areDiagsSelfContained();
        }
        Diag << FixItHint::CreateRemoval(
            CharSourceRange::getCharRange(S->getBeginLoc(), SemiColonEnd));
      }
    }
  }
}

} // namespace clang::tidy::cppcoreguidelines
