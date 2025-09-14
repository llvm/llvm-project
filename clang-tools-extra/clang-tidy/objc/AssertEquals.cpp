//===----------------------------------------------------------------------===//
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
static const std::map<std::string, std::string> &nameMap() {
  static std::map<std::string, std::string> Map{
      {"XCTAssertEqual", "XCTAssertEqualObjects"},
      {"XCTAssertNotEqual", "XCTAssertNotEqualObjects"},

  };
  return Map;
}

void AssertEquals::registerMatchers(MatchFinder *Finder) {
  for (const auto &Pair : nameMap()) {
    Finder->addMatcher(
        binaryOperator(anyOf(hasOperatorName("!="), hasOperatorName("==")),
                       isExpandedFromMacro(Pair.first),
                       anyOf(hasLHS(hasType(qualType(
                                 hasCanonicalType(asString("NSString *"))))),
                             hasRHS(hasType(qualType(
                                 hasCanonicalType(asString("NSString *"))))))

                           )
            .bind(Pair.first),
        this);
  }
}

void AssertEquals::check(const ast_matchers::MatchFinder::MatchResult &Result) {
  for (const auto &Pair : nameMap()) {
    if (const auto *Root = Result.Nodes.getNodeAs<BinaryOperator>(Pair.first)) {
      SourceManager *Sm = Result.SourceManager;
      // The macros are nested two levels, so going up twice.
      auto MacroCallsite = Sm->getImmediateMacroCallerLoc(
          Sm->getImmediateMacroCallerLoc(Root->getBeginLoc()));
      diag(MacroCallsite, "use " + Pair.second + " for comparing objects")
          << FixItHint::CreateReplacement(
                 clang::CharSourceRange::getCharRange(
                     MacroCallsite,
                     MacroCallsite.getLocWithOffset(Pair.first.length())),
                 Pair.second);
    }
  }
}

} // namespace clang::tidy::objc
