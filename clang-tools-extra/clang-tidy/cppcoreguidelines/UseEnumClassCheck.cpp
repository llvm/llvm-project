//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseEnumClassCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

UseEnumClassCheck::UseEnumClassCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreUnscopedEnumsInClasses(
          Options.get("IgnoreUnscopedEnumsInClasses", false)) {}

void UseEnumClassCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreUnscopedEnumsInClasses",
                IgnoreUnscopedEnumsInClasses);
}

void UseEnumClassCheck::registerMatchers(MatchFinder *Finder) {
  auto EnumDecl =
      IgnoreUnscopedEnumsInClasses
          ? enumDecl(unless(isScoped()), unless(hasParent(recordDecl())))
          : enumDecl(unless(isScoped()));
  Finder->addMatcher(EnumDecl.bind("unscoped_enum"), this);
}

void UseEnumClassCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *UnscopedEnum = Result.Nodes.getNodeAs<EnumDecl>("unscoped_enum");

  diag(UnscopedEnum->getLocation(),
       "enum %0 is unscoped, use 'enum class' instead")
      << UnscopedEnum;
}

} // namespace clang::tidy::cppcoreguidelines
