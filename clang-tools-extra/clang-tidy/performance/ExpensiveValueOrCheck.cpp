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

namespace {

bool hasOperatorStar(const CXXRecordDecl *RD) {
  return llvm::any_of(RD->methods(), [](const CXXMethodDecl *M) {
    return M->getOverloadedOperator() == OO_Star;
  });
}

StringRef findValueMethod(const CXXRecordDecl *RD) {
  for (const auto *M : RD->methods()) {
    if (!M->getDeclName().isIdentifier())
      continue;
    StringRef Name = M->getName();
    if (Name == "value" || Name == "Value")
      return Name;
  }
  return {};
}

std::string buildSuggestion(const CXXRecordDecl *OptionalClass) {
  bool HasDeref = hasOperatorStar(OptionalClass);
  StringRef ValueName = findValueMethod(OptionalClass);

  if (HasDeref && !ValueName.empty())
    return (llvm::Twine("consider using 'operator*' or '") + ValueName +
            "()' with a separate fallback")
        .str();
  if (HasDeref)
    return "consider using 'operator*' with a separate fallback";
  if (!ValueName.empty())
    return (llvm::Twine("consider using '") + ValueName +
            "()' with a separate fallback")
        .str();
  return "consider avoiding the copy";
}

} // namespace

ExpensiveValueOrCheck::ExpensiveValueOrCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SizeThreshold(Options.get("SizeThreshold", 16U)),
      OptionalTypes(utils::options::parseStringList(
          Options.get("OptionalTypes",
                      "::std::optional;::absl::optional;::boost::optional"))) {}

void ExpensiveValueOrCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SizeThreshold", SizeThreshold);
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

  const ASTContext &Ctx = *Result.Context;
  const QualType ValueType = Call->getType().getCanonicalType();

  // Rvalue optional uses && overload which moves. Suppress if move is cheap.
  if (ObjExpr && !ObjExpr->isLValue() &&
      utils::type_traits::hasNonTrivialMoveConstructor(ValueType))
    return;

  const bool IsExpensiveType =
      utils::type_traits::isExpensiveToCopy(ValueType, Ctx).value_or(false);
  const int64_t ValueSize = Ctx.getTypeSizeInChars(ValueType).getQuantity();
  const bool IsExpensive =
      IsExpensiveType || ValueSize > static_cast<int64_t>(SizeThreshold);

  if (!IsExpensive)
    return;

  const CXXMethodDecl *Method = Call->getMethodDecl();

  diag(Call->getExprLoc(), "'%0' copies expensive type %1; %2")
      << Method->getName() << ValueType
      << buildSuggestion(Method->getParent());

  const Expr *FallbackArg = Call->getArg(0)->IgnoreImplicit();
  if (FallbackArg->HasSideEffects(Ctx))
    diag(FallbackArg->getExprLoc(),
         "the fallback is always evaluated; a conditional rewrite would "
         "change evaluation semantics",
         DiagnosticIDs::Note);
}

} // namespace clang::tidy::performance
