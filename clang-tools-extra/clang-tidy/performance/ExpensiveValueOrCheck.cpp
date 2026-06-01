//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpensiveValueOrCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

ExpensiveValueOrCheck::ExpensiveValueOrCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SizeThreshold(Options.get("SizeThreshold", 16U)),
      WarnOnRvalueOptional(Options.get("WarnOnRvalueOptional", false)),
      OptionalTypes(utils::options::parseStringList(
          Options.get("OptionalTypes",
                      "::std::optional;::absl::optional;::boost::optional"))) {}

void ExpensiveValueOrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SizeThreshold", SizeThreshold);
  Options.store(Opts, "WarnOnRvalueOptional", WarnOnRvalueOptional);
  Options.store(Opts, "OptionalTypes",
                utils::options::serializeStringList(OptionalTypes));
}

void ExpensiveValueOrCheck::registerMatchers(MatchFinder *Finder) {
  auto OptionalTypesMatcher =
      matchers::matchesAnyListedRegexName(OptionalTypes);
  auto ValueOrMatcher = hasAnyName("value_or", "valueOr", "ValueOr");

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxMethodDecl(ValueOrMatcher, ofClass(OptionalTypesMatcher))))
          .bind("call"),
      this);
}

void ExpensiveValueOrCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXMemberCallExpr>("call");
  if (!Call)
    return;

  const Expr *ObjExpr = Call->getImplicitObjectArgument();
  if (!WarnOnRvalueOptional && ObjExpr && !ObjExpr->isLValue())
    return;

  const ASTContext &Ctx = *Result.Context;
  const QualType ValueType = Call->getType().getCanonicalType();
  const bool IsExpensiveType =
      utils::type_traits::isExpensiveToCopy(ValueType, Ctx).value_or(false);

  const int64_t ValueSize = Ctx.getTypeSizeInChars(ValueType).getQuantity();
  const bool IsExpensive =
      IsExpensiveType || ValueSize > static_cast<int64_t>(SizeThreshold);

  if (!IsExpensive)
    return;

  const CXXMethodDecl *Method = Call->getMethodDecl();

  diag(Call->getExprLoc(),
       "'%0' copies expensive type %1; consider using 'operator*' or "
       "'value()' with a separate fallback")
      << Method->getName() << ValueType;
}

} // namespace clang::tidy::performance
