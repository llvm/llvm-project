//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnorderedEqualCompareCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

UnorderedEqualCompareCheck::UnorderedEqualCompareCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ContainerNames(utils::options::parseStringList(Options.get(
          "Containers", "::std::unordered_set;::std::unordered_multiset;"
                        "::std::unordered_map;::std::unordered_multimap"))) {}

void UnorderedEqualCompareCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "Containers",
                utils::options::serializeStringList(ContainerNames));
}

void UnorderedEqualCompareCheck::registerMatchers(MatchFinder *Finder) {
  auto UnorderedContainer = hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(hasAnyName(ContainerNames))))));

  // An iterator obtained from an unordered container, e.g. c.begin().
  auto IteratorFromUnordered = cxxMemberCallExpr(
      callee(cxxMethodDecl(hasAnyName("begin", "end", "cbegin", "cend"))),
      on(expr(UnorderedContainer)));

  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::std::equal"))),
                              hasAnyArgument(IteratorFromUnordered))
                         .bind("call"),
                     this);
}

void UnorderedEqualCompareCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>("call");
  diag(Call->getBeginLoc(),
       "comparing an unordered container with 'std::equal' is order-dependent");
}

} // namespace clang::tidy::bugprone
