//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PreferIsaOrDynCastInConditionalsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

namespace {
AST_MATCHER(Expr, isMacroID) { return Node.getExprLoc().isMacroID(); }
} // namespace

void PreferIsaOrDynCastInConditionalsCheck::registerMatchers(
    MatchFinder *Finder) {
  auto AnyCalleeName = [](ArrayRef<StringRef> CalleeName) {
    return allOf(unless(isMacroID()), unless(cxxMemberCallExpr()),
                 callee(expr(ignoringImpCasts(
                     declRefExpr(to(namedDecl(hasAnyName(CalleeName))),
                                 hasAnyTemplateArgumentLoc(anything()))
                         .bind("callee")))));
  };

  auto CondExpr = hasCondition(implicitCastExpr(
      has(callExpr(AnyCalleeName({"cast", "dyn_cast"})).bind("cond"))));

  auto CondExprOrCondVar =
      anyOf(hasConditionVariableStatement(containsDeclaration(
                0, varDecl(hasInitializer(callExpr(AnyCalleeName({"cast"}))))
                       .bind("var"))),
            CondExpr);

  auto CallWithBindedArg =
      callExpr(
          AnyCalleeName(
              {"isa", "cast", "cast_or_null", "dyn_cast", "dyn_cast_or_null"}),
          hasArgument(0, mapAnyOf(declRefExpr, cxxMemberCallExpr).bind("arg")))
          .bind("rhs");

  Finder->addMatcher(
      stmt(anyOf(ifStmt(CondExprOrCondVar), forStmt(CondExprOrCondVar),
                 whileStmt(CondExprOrCondVar), doStmt(CondExpr),
                 binaryOperator(unless(isExpansionInFileMatching(
                                    "llvm/include/llvm/Support/Casting.h")),
                                hasOperatorName("&&"),
                                hasLHS(implicitCastExpr().bind("lhs")),
                                hasRHS(ignoringImpCasts(CallWithBindedArg)))
                     .bind("and"))),
      this);
}

void PreferIsaOrDynCastInConditionalsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Callee = Result.Nodes.getNodeAs<DeclRefExpr>("callee");

  assert(Callee && "Callee should be binded if anything is matched");

  // The first and last letter of the identifier
  //   llvm::cast<T>(x)
  //         ^  ^
  //  StartLoc  EndLoc
  SourceLocation StartLoc = Callee->getLocation();
  SourceLocation EndLoc = Callee->getNameInfo().getEndLoc();

  if (Result.Nodes.getNodeAs<VarDecl>("var")) {
    diag(StartLoc,
         "cast<> in conditional will assert rather than return a null pointer")
        << FixItHint::CreateReplacement(SourceRange(StartLoc, EndLoc),
                                        "dyn_cast");
  } else if (Result.Nodes.getNodeAs<CallExpr>("cond")) {
    StringRef Message =
        "cast<> in conditional will assert rather than return a null pointer";
    if (Callee->getDecl()->getName() == "dyn_cast")
      Message = "return value from dyn_cast<> not used";

    diag(StartLoc, Message)
        << FixItHint::CreateReplacement(SourceRange(StartLoc, EndLoc), "isa");
  } else if (Result.Nodes.getNodeAs<BinaryOperator>("and")) {
    const auto *LHS = Result.Nodes.getNodeAs<ImplicitCastExpr>("lhs");
    const auto *RHS = Result.Nodes.getNodeAs<CallExpr>("rhs");
    const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg");

    assert(LHS && "LHS is null");
    assert(RHS && "RHS is null");
    assert(Arg && "Arg is null");

    auto GetText = [&](SourceRange R) {
      return Lexer::getSourceText(CharSourceRange::getTokenRange(R),
                                  *Result.SourceManager, getLangOpts());
    };

    const StringRef LHSString = GetText(LHS->getSourceRange());
    const StringRef ArgString = GetText(Arg->getSourceRange());

    if (ArgString != LHSString)
      return;

    // It is not clear which is preferred between `isa_and_nonnull` and
    // `isa_and_present`. See
    // https://discourse.llvm.org/t/psa-swapping-out-or-null-with-if-present/65018
    const std::string Replacement = llvm::formatv(
        "{}isa_and_nonnull{}",
        GetText(Callee->getQualifierLoc().getSourceRange()),
        GetText(SourceRange(Callee->getLAngleLoc(), RHS->getEndLoc())));

    diag(LHS->getBeginLoc(),
         "isa_and_nonnull<> is preferred over an explicit test for null "
         "followed by calling isa<>")
        << FixItHint::CreateReplacement(
               SourceRange(LHS->getBeginLoc(), RHS->getEndLoc()), Replacement);
  } else {
    llvm_unreachable(
        R"(One of "var", "cond" and "and" should be binded if anything is matched)");
  }
}

} // namespace clang::tidy::llvm_check
