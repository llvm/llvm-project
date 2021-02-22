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

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

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

static const std::pair<const FieldDecl *, const Expr *>
isAssignmentToMemberOf(const RecordDecl *Rec, const Stmt *S) {
  if (const auto *BO = dyn_cast<BinaryOperator>(S)) {
    if (BO->getOpcode() != BO_Assign)
      return std::make_pair(nullptr, nullptr);

    const auto *ME = dyn_cast<MemberExpr>(BO->getLHS()->IgnoreParenImpCasts());
    if (!ME)
      return std::make_pair(nullptr, nullptr);

    const auto *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
    if (!Field)
      return std::make_pair(nullptr, nullptr);

    if (isa<CXXThisExpr>(ME->getBase()))
      return std::make_pair(Field, BO->getRHS()->IgnoreParenImpCasts());
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

    if (isa<CXXThisExpr>(ME->getBase()))
      return std::make_pair(Field, COCE->getArg(1)->IgnoreParenImpCasts());
  }

  return std::make_pair(nullptr, nullptr);
}

PreferMemberInitializerCheck::PreferMemberInitializerCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IsUseDefaultMemberInitEnabled(
          Context->isCheckEnabled("modernize-use-default-member-init")),
      UseAssignment(OptionsView("modernize-use-default-member-init",
                                Context->getOptions().CheckOptions, Context)
                        .get("UseAssignment", false)) {}

void PreferMemberInitializerCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UseAssignment", UseAssignment);
}

void PreferMemberInitializerCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxConstructorDecl(hasBody(compoundStmt()), unless(isInstantiated()))
          .bind("ctor"),
      this);
}

void PreferMemberInitializerCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  const auto *Body = cast<CompoundStmt>(Ctor->getBody());

  const CXXRecordDecl *Class = Ctor->getParent();
  SourceLocation InsertPos;
  bool FirstToCtorInits = true;

  for (const Stmt *S : Body->body()) {
    if (S->getBeginLoc().isMacroID()) {
      StringRef MacroName =
        Lexer::getImmediateMacroName(S->getBeginLoc(), *Result.SourceManager,
                                     getLangOpts());
      if (MacroName.contains_lower("assert"))
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

    const FieldDecl *Field;
    const Expr *InitValue;
    std::tie(Field, InitValue) = isAssignmentToMemberOf(Class, S);
    if (Field) {
      if (IsUseDefaultMemberInitEnabled && getLangOpts().CPlusPlus11 &&
          Ctor->isDefaultConstructor() &&
          (getLangOpts().CPlusPlus20 || !Field->isBitField()) &&
          (!isa<RecordDecl>(Class->getDeclContext()) ||
           !cast<RecordDecl>(Class->getDeclContext())->isUnion()) &&
          shouldBeDefaultMemberInitializer(InitValue)) {

        SourceLocation FieldEnd =
            Lexer::getLocForEndOfToken(Field->getSourceRange().getEnd(), 0,
                                       *Result.SourceManager, getLangOpts());
        SmallString<128> Insertion(
            {UseAssignment ? " = " : "{",
             Lexer::getSourceText(
                 CharSourceRange(InitValue->getSourceRange(), true),
                 *Result.SourceManager, getLangOpts()),
             UseAssignment ? "" : "}"});

        SourceLocation SemiColonEnd =
            Lexer::findNextToken(S->getEndLoc(), *Result.SourceManager,
                                 getLangOpts())
                ->getEndLoc();
        CharSourceRange StmtRange =
            CharSourceRange::getCharRange(S->getBeginLoc(), SemiColonEnd);

        diag(S->getBeginLoc(), "%0 should be initialized in an in-class"
                               " default member initializer")
            << Field << FixItHint::CreateInsertion(FieldEnd, Insertion)
            << FixItHint::CreateRemoval(StmtRange);
      } else {
        SmallString<128> Insertion;
        bool AddComma = false;
        if (!Ctor->getNumCtorInitializers() && FirstToCtorInits) {
          SourceLocation BodyPos = Ctor->getBody()->getBeginLoc();
          SourceLocation NextPos = Ctor->getBeginLoc();
          do {
            InsertPos = NextPos;
            NextPos = Lexer::findNextToken(NextPos, *Result.SourceManager,
                                           getLangOpts())
                          ->getLocation();
          } while (NextPos != BodyPos);
          InsertPos = Lexer::getLocForEndOfToken(
              InsertPos, 0, *Result.SourceManager, getLangOpts());

          Insertion = " : ";
        } else {
          bool Found = false;
          unsigned Index = Field->getFieldIndex();
          for (const auto *Init : Ctor->inits()) {
            if (Init->isMemberInitializer()) {
              if (Index < Init->getMember()->getFieldIndex()) {
                InsertPos = Init->getSourceLocation();
                Found = true;
                break;
              }
            }
          }

          if (!Found) {
            if (Ctor->getNumCtorInitializers()) {
              InsertPos = Lexer::getLocForEndOfToken(
                  (*Ctor->init_rbegin())->getSourceRange().getEnd(), 0,
                  *Result.SourceManager, getLangOpts());
            }
            Insertion = ", ";
          } else {
            AddComma = true;
          }
        }
        Insertion.append(
            {Field->getName(), "(",
             Lexer::getSourceText(
                 CharSourceRange(InitValue->getSourceRange(), true),
                 *Result.SourceManager, getLangOpts()),
             AddComma ? "), " : ")"});

        SourceLocation SemiColonEnd =
            Lexer::findNextToken(S->getEndLoc(), *Result.SourceManager,
                                 getLangOpts())
                ->getEndLoc();
        CharSourceRange StmtRange =
            CharSourceRange::getCharRange(S->getBeginLoc(), SemiColonEnd);

        diag(S->getBeginLoc(), "%0 should be initialized in a member"
                               " initializer of the constructor")
            << Field
            << FixItHint::CreateInsertion(InsertPos, Insertion,
                                          FirstToCtorInits)
            << FixItHint::CreateRemoval(StmtRange);
        FirstToCtorInits = false;
      }
    }
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
