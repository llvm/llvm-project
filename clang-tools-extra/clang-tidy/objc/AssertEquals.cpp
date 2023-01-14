//===--- AssertEquals.cpp - clang-tidy --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssertEquals.h"

#include <map>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

// Mapping from `XCTAssert*Equal` to `XCTAssert*EqualObjects` name.
static const std::map<std::string, std::string> &NameMap() {
  static std::map<std::string, std::string> map{
      {"XCTAssertEqual", "XCTAssertEqualObjects"},
      {"XCTAssertNotEqual", "XCTAssertNotEqualObjects"},

  };
  return map;
}

void AssertEquals::registerMatchers(MatchFinder *finder) {
  for (const auto &pair : NameMap()) {
    finder->addMatcher(
        binaryOperator(anyOf(hasOperatorName("!="), hasOperatorName("==")),
                       isExpandedFromMacro(pair.first),
                       anyOf(hasLHS(hasType(qualType(
                                 hasCanonicalType(asString("NSString *"))))),
                             hasRHS(hasType(qualType(
                                 hasCanonicalType(asString("NSString *"))))))

                           )
            .bind(pair.first),
        this);
  }
}

void AssertEquals::check(const ast_matchers::MatchFinder::MatchResult &result) {
  for (const auto &pair : NameMap()) {
    if (const auto *root = result.Nodes.getNodeAs<BinaryOperator>(pair.first)) {
      SourceManager *sm = result.SourceManager;
      // The macros are nested two levels, so going up twice.
      auto macro_callsite = sm->getImmediateMacroCallerLoc(
          sm->getImmediateMacroCallerLoc(root->getBeginLoc()));
      diag(macro_callsite, "use " + pair.second + " for comparing objects")
          << FixItHint::CreateReplacement(
                 clang::CharSourceRange::getCharRange(
                     macro_callsite,
                     macro_callsite.getLocWithOffset(pair.first.length())),
                 pair.second);
    }
  }
}

} // namespace clang::tidy::objc
