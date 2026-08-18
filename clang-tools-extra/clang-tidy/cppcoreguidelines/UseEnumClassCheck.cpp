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
using namespace clang::ast_matchers::internal;

namespace clang::tidy::cppcoreguidelines {
namespace {
// FIXME: The matcher 'hasName(Name)' asserts that its argument 'Name' is
// nonempty. Perhaps remove that assertion and replace 'isUnnamed()' with
// 'hasName("")'.
AST_MATCHER(EnumDecl, isUnnamed) { return Node.getName().empty(); }
} // namespace

UseEnumClassCheck::UseEnumClassCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IgnoreUnscopedEnumsInClasses(
          Options.get("IgnoreUnscopedEnumsInClasses", false)),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void UseEnumClassCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreUnscopedEnumsInClasses",
                IgnoreUnscopedEnumsInClasses);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void UseEnumClassCheck::registerMatchers(MatchFinder *Finder) {
  const auto EnumDecl = IgnoreUnscopedEnumsInClasses
                            ? enumDecl(unless(isScoped()), unless(isUnnamed()),
                                       unless(hasParent(recordDecl())))
                            : enumDecl(unless(isScoped()), unless(isUnnamed()));
  Finder->addMatcher(EnumDecl.bind("unscoped_enum"), this);
}

void UseEnumClassCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *UnscopedEnum = Result.Nodes.getNodeAs<EnumDecl>("unscoped_enum");
  const SourceLocation SourceLoc = UnscopedEnum->getLocation();

  if (IgnoreMacros && SourceLoc.isMacroID())
    return;

  diag(SourceLoc, "enum %0 is unscoped, use 'enum class' instead")
      << UnscopedEnum;
}

} // namespace clang::tidy::cppcoreguidelines
