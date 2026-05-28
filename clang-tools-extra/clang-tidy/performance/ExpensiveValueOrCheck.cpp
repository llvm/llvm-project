//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpensiveValueOrCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

ExpensiveValueOrCheck::ExpensiveValueOrCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SizeThreshold(Options.get("SizeThreshold", 8U)),
      WarnOnRvalueOptional(Options.get("WarnOnRvalueOptional", false)),
      OptionalTypes(utils::options::parseStringList(
          Options.get("OptionalTypes", "::std::optional"))) {}

void ExpensiveValueOrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SizeThreshold", SizeThreshold);
  Options.store(Opts, "WarnOnRvalueOptional", WarnOnRvalueOptional);
  Options.store(Opts, "OptionalTypes",
                utils::options::serializeStringList(OptionalTypes));
}

void ExpensiveValueOrCheck::registerMatchers(MatchFinder *Finder) {
  auto OptionalTypesMatcher = hasAnyName(OptionalTypes);

  Finder->addMatcher(
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("value_or"),
                                             ofClass(OptionalTypesMatcher))))
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

  const QualType ValueType = Call->getType().getCanonicalType();
  if (ValueType->isDependentType() || ValueType->isIncompleteType())
    return;

  const ASTContext &Ctx = *Result.Context;
  const int64_t ValueSize = Ctx.getTypeSizeInChars(ValueType).getQuantity();
  const bool IsExpensive = !ValueType.isTriviallyCopyableType(Ctx) ||
                           ValueSize > static_cast<int64_t>(SizeThreshold);

  if (!IsExpensive)
    return;

  diag(Call->getExprLoc(),
       "'value_or' copies expensive type %0; consider using 'operator*' or "
       "'value()' with a separate fallback")
      << ValueType;
}

} // namespace clang::tidy::performance
