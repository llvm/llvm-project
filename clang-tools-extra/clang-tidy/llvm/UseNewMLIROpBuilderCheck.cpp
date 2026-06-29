//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseNewMLIROpBuilderCheck.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void UseNewMlirOpBuilderCheck::registerMatchers(MatchFinder *Finder) {
  // Match a create call on an OpBuilder.
  const auto BuilderType =
      cxxRecordDecl(isSameOrDerivedFrom("::mlir::OpBuilder"));
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxMethodDecl(hasTemplateArgument(0, templateArgument()),
                               hasName("create"))),
          on(expr(anyOf(hasType(BuilderType), hasType(pointsTo(BuilderType))))
                 .bind("builder")))
          .bind("call"),
      this);
}

void UseNewMlirOpBuilderCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  const auto *Builder = Result.Nodes.getNodeAs<Expr>("builder");

  const DiagnosticBuilder Diag =
      diag(Result.SourceManager->getExpansionLoc(Call->getBeginLoc()),
           "use 'OpType::create(builder, ...)' instead of "
           "'builder.create<OpType>(...)'");

  if (Call->getBeginLoc().isMacroID())
    return;

  // Only attempt the rewrite if given an lvalue builder.
  if (isa<CXXTemporaryObjectExpr>(Builder))
    return;

  const auto *Callee = dyn_cast<MemberExpr>(Call->getCallee());
  if (!Callee)
    return;

  Diag << FixItHint::CreateRemoval(
              {Call->getBeginLoc(), Callee->getLAngleLoc()})
       << FixItHint::CreateReplacement(Callee->getRAngleLoc(), "::create");

  std::string BuilderArg;
  if (Builder->isImplicitCXXThis()) {
    BuilderArg = "*this";
  } else {
    if (Callee->isArrow())
      BuilderArg += '*';

    BuilderArg += Lexer::getSourceText(
        CharSourceRange::getTokenRange(Builder->getSourceRange()),
        *Result.SourceManager, getLangOpts());
  }

  if (Call->getNumArgs() == 0) {
    Diag << FixItHint::CreateInsertion(Call->getRParenLoc(), BuilderArg);
  } else {
    BuilderArg += ", ";
    Diag << FixItHint::CreateInsertion(Call->getArg(0)->getBeginLoc(),
                                       BuilderArg);
  }
}

} // namespace clang::tidy::llvm_check
