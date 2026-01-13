//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AssertEqualsCheck.h"
#include "llvm/ADT/StringMap.h"

#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

// Mapping from `XCTAssert*Equal` to `XCTAssert*EqualObjects` name.
static const llvm::StringMap<StringRef> NameMap{
    {"XCTAssertEqual", "XCTAssertEqualObjects"},
    {"XCTAssertNotEqual", "XCTAssertNotEqualObjects"},
};

void AssertEqualsCheck::registerMatchers(MatchFinder *Finder) {
  for (const auto &[CurrName, _] : NameMap) {
    Finder->addMatcher(
        binaryOperator(anyOf(hasOperatorName("!="), hasOperatorName("==")),
                       isExpandedFromMacro(std::string(CurrName)),
                       anyOf(hasLHS(hasType(qualType(
                                 hasCanonicalType(asString("NSString *"))))),
                             hasRHS(hasType(qualType(
                                 hasCanonicalType(asString("NSString *")))))))
            .bind(CurrName),
        this);
  }
}

void AssertEqualsCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  for (const auto &[CurrName, TargetName] : NameMap) {
    if (const auto *Root = Result.Nodes.getNodeAs<BinaryOperator>(CurrName)) {
      const SourceManager *Sm = Result.SourceManager;
      // The macros are nested two levels, so going up twice.
      auto MacroCallsite = Sm->getImmediateMacroCallerLoc(
          Sm->getImmediateMacroCallerLoc(Root->getBeginLoc()));
      diag(MacroCallsite,
           (Twine("use ") + TargetName + " for comparing objects").str())
          << FixItHint::CreateReplacement(
                 clang::CharSourceRange::getCharRange(
                     MacroCallsite,
                     MacroCallsite.getLocWithOffset(CurrName.size())),
                 TargetName);
    }
  }
}

} // namespace clang::tidy::objc
