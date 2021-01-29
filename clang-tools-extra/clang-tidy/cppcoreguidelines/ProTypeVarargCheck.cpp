//===--- ProTypeVarargCheck.cpp - clang-tidy-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProTypeVarargCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

const internal::VariadicDynCastAllOfMatcher<Stmt, VAArgExpr> VAArgExpr;

static constexpr StringRef AllowedVariadics[] = {
    // clang-format off
    "__builtin_isgreater", 
    "__builtin_isgreaterequal", 
    "__builtin_isless",
    "__builtin_islessequal", 
    "__builtin_islessgreater", 
    "__builtin_isunordered",
    "__builtin_fpclassify", 
    "__builtin_isfinite", 
    "__builtin_isinf",
    "__builtin_isinf_sign", 
    "__builtin_isnan", 
    "__builtin_isnormal",
    "__builtin_signbit", 
    "__builtin_constant_p", 
    "__builtin_classify_type",
    "__builtin_va_start",
    "__builtin_assume_aligned", // Documented as variadic to support default 
                                // parameters.
    "__builtin_prefetch",       // Documented as variadic to support default
                                // parameters.
    "__builtin_shufflevector",  // Documented as variadic but with a defined
                                // number of args based on vector size.
    "__builtin_convertvector", 
    "__builtin_call_with_static_chain",
    "__builtin_annotation", 
    "__builtin_add_overflow", 
    "__builtin_sub_overflow",
    "__builtin_mul_overflow", 
    "__builtin_preserve_access_index",
    "__builtin_nontemporal_store", 
    "__builtin_nontemporal_load",
    "__builtin_ms_va_start",
    // clang-format on
};

namespace {
AST_MATCHER(QualType, isVAList) {
  ASTContext &Context = Finder->getASTContext();
  QualType Desugar = Node.getDesugaredType(Context);
  return Context.getBuiltinVaListType().getDesugaredType(Context) == Desugar ||
         Context.getBuiltinMSVaListType().getDesugaredType(Context) == Desugar;
}

AST_MATCHER_P(AdjustedType, hasOriginalType,
              ast_matchers::internal::Matcher<QualType>, InnerType) {
  return InnerType.matches(Node.getOriginalType(), Finder, Builder);
}
} // namespace

void ProTypeVarargCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(VAArgExpr().bind("va_use"), this);

  Finder->addMatcher(
      callExpr(callee(functionDecl(isVariadic(),
                                   unless(hasAnyName(AllowedVariadics)))))
          .bind("callvararg"),
      this);

  Finder->addMatcher(
      varDecl(unless(parmVarDecl()),
              hasType(qualType(
                  anyOf(isVAList(), decayedType(hasOriginalType(isVAList()))))))
          .bind("va_list"),
      this);
}

static bool hasSingleVariadicArgumentWithValue(const CallExpr *C, uint64_t I) {
  const auto *FDecl = dyn_cast<FunctionDecl>(C->getCalleeDecl());
  if (!FDecl)
    return false;

  auto N = FDecl->getNumParams(); // Number of parameters without '...'
  if (C->getNumArgs() != N + 1)
    return false; // more/less than one argument passed to '...'

  const auto *IntLit =
      dyn_cast<IntegerLiteral>(C->getArg(N)->IgnoreParenImpCasts());
  if (!IntLit)
    return false;

  if (IntLit->getValue() != I)
    return false;

  return true;
}

void ProTypeVarargCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Matched = Result.Nodes.getNodeAs<CallExpr>("callvararg")) {
    if (hasSingleVariadicArgumentWithValue(Matched, 0))
      return;
    diag(Matched->getExprLoc(), "do not call c-style vararg functions");
  }

  if (const auto *Matched = Result.Nodes.getNodeAs<Expr>("va_use")) {
    diag(Matched->getExprLoc(),
         "do not use va_arg to define c-style vararg functions; "
         "use variadic templates instead");
  }

  if (const auto *Matched = Result.Nodes.getNodeAs<VarDecl>("va_list")) {
    auto SR = Matched->getSourceRange();
    if (SR.isInvalid())
      return; // some implicitly generated builtins take va_list
    diag(SR.getBegin(), "do not declare variables of type va_list; "
                        "use variadic templates instead");
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
