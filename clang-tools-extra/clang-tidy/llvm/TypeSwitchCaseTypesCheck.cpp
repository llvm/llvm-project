//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeSwitchCaseTypesCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void TypeSwitchCaseTypesCheck::registerMatchers(MatchFinder *Finder) {
  // Match calls to `llvm::TypeSwitch::Case` with a lambda expression.
  // Explicit template arguments and their count are checked in `check()`.
  Finder->addMatcher(
      cxxMemberCallExpr(
          argumentCountIs(1),
          callee(memberExpr(member(cxxMethodDecl(hasName("Case"),
                                                 ofClass(cxxRecordDecl(hasName(
                                                     "::llvm::TypeSwitch"))))))
                     .bind("member")),
          hasArgument(0, lambdaExpr().bind("lambda")))
          .bind("call"),
      this);
}

void TypeSwitchCaseTypesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  assert(Call);
  const auto *Lambda = Result.Nodes.getNodeAs<LambdaExpr>("lambda");
  assert(Lambda);
  const auto *MemExpr = Result.Nodes.getNodeAs<MemberExpr>("member");
  assert(MemExpr);

  // Only handle `Case<T>` with exactly one explicit template argument.
  if (!MemExpr->hasExplicitTemplateArgs() || MemExpr->getNumTemplateArgs() != 1)
    return;

  const TemplateArgumentLoc &TemplateArg = MemExpr->getTemplateArgs()[0];
  if (TemplateArg.getArgument().getKind() != TemplateArgument::Type)
    return;

  // Get the lambda's call operator to examine its parameter.
  const CXXMethodDecl *CallOp = Lambda->getCallOperator();
  if (!CallOp || CallOp->getNumParams() != 1)
    return;

  const ParmVarDecl *LambdaParam = CallOp->getParamDecl(0);
  const QualType ParamType = LambdaParam->getType();

  // Check if the parameter uses `auto`.
  QualType ParamBaseType = ParamType.getNonReferenceType();
  while (ParamBaseType->isPointerType())
    ParamBaseType = ParamBaseType->getPointeeType();
  const bool ParamIsAuto = ParamBaseType->getUnqualifiedDesugaredType()
                               ->getAs<TemplateTypeParmType>() != nullptr;

  if (ParamIsAuto) {
    // Warn about `.Case<T>([](auto x) {...})` -- prefer explicit lambda
    // parameter type. We only emit a warning without a fixit because we cannot
    // reliably determine the deduced type of `auto`. The actual type depends on
    // how `dyn_cast<CaseT>` behaves for the `TypeSwitch` value type, which
    // varies (e.g., pointer types return pointers, but MLIR handle types may
    // return by value).
    diag(Call->getExprLoc(),
         "lambda parameter needlessly uses 'auto', use explicit type instead");
    diag(LambdaParam->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
         "replace 'auto' with explicit type", DiagnosticIDs::Note);
    diag(TemplateArg.getLocation(),
         "type from template argument can be inferred and removed",
         DiagnosticIDs::Note);
    return;
  }

  // Handle `.Case<T>([](T x) {...})` -> `.Case([](T x) {...})`.
  // Only warn if the types match (otherwise it might be intentional or a bug).
  const QualType CaseType = TemplateArg.getArgument().getAsType();
  if (CaseType->getCanonicalTypeUnqualified() !=
      ParamBaseType->getCanonicalTypeUnqualified())
    return;

  auto Diag = diag(Call->getExprLoc(), "redundant explicit template argument");

  // Skip fixit if template argument involves macros.
  const SourceLocation LAngleLoc = MemExpr->getLAngleLoc();
  const SourceLocation RAngleLoc = MemExpr->getRAngleLoc();
  if (LAngleLoc.isInvalid() || RAngleLoc.isInvalid() || LAngleLoc.isMacroID() ||
      RAngleLoc.isMacroID())
    return;

  Diag << FixItHint::CreateRemoval(SourceRange(LAngleLoc, RAngleLoc));

  // Also remove `template` keyword, if present.
  if (MemExpr->hasTemplateKeyword())
    Diag << FixItHint::CreateRemoval(MemExpr->getTemplateKeywordLoc());
}

} // namespace clang::tidy::llvm_check
