//===--- Transformer.cpp - Transformer library implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/Transformer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "llvm/Support/Error.h"
#include <utility>
#include <vector>

using namespace clang;
using namespace tooling;

using ast_matchers::MatchFinder;

void Transformer::registerMatchers(MatchFinder *MatchFinder) {
  for (auto &Matcher : transformer::detail::buildMatchers(Rule))
    MatchFinder->addDynamicMatcher(Matcher, this);
}

void Transformer::run(const MatchFinder::MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasErrorOccurred())
    return;

  transformer::RewriteRule::Case Case =
      transformer::detail::findSelectedCase(Result, Rule);
  auto Transformations = transformer::detail::translateEdits(Result, Case.Edits);
  if (!Transformations) {
    Consumer(Transformations.takeError());
    return;
  }

  if (Transformations->empty()) {
    // No rewrite applied (but no error encountered either).
    transformer::detail::getRuleMatchLoc(Result).print(
        llvm::errs() << "note: skipping match at loc ", *Result.SourceManager);
    llvm::errs() << "\n";
    return;
  }

  // Record the results in the AtomicChange, anchored at the location of the
  // first change.
  AtomicChange AC(*Result.SourceManager,
                  (*Transformations)[0].Range.getBegin());
  for (const auto &T : *Transformations) {
    if (auto Err = AC.replace(*Result.SourceManager, T.Range, T.Replacement)) {
      Consumer(std::move(Err));
      return;
    }
  }

  for (const auto &I : Case.AddedIncludes) {
    auto &Header = I.first;
    switch (I.second) {
    case transformer::IncludeFormat::Quoted:
      AC.addHeader(Header);
      break;
    case transformer::IncludeFormat::Angled:
      AC.addHeader((llvm::Twine("<") + Header + ">").str());
      break;
    }
  }

  Consumer(std::move(AC));
}
