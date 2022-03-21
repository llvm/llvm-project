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
#include <map>
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
  auto Transformations = Case.Edits(Result);
  if (!Transformations) {
    Consumer(Transformations.takeError());
    return;
  }

  if (Transformations->empty())
    return;

  // Group the transformations, by file, into AtomicChanges, each anchored by
  // the location of the first change in that file.
  std::map<FileID, AtomicChange> ChangesByFileID;
  for (const auto &T : *Transformations) {
    auto ID = Result.SourceManager->getFileID(T.Range.getBegin());
    auto Iter = ChangesByFileID
                    .emplace(ID, AtomicChange(*Result.SourceManager,
                                              T.Range.getBegin(), T.Metadata))
                    .first;
    auto &AC = Iter->second;
    switch (T.Kind) {
    case transformer::EditKind::Range:
      if (auto Err =
              AC.replace(*Result.SourceManager, T.Range, T.Replacement)) {
        Consumer(std::move(Err));
        return;
      }
      break;
    case transformer::EditKind::AddInclude:
      AC.addHeader(T.Replacement);
      break;
    }
  }

  llvm::SmallVector<AtomicChange, 1> Changes;
  Changes.reserve(ChangesByFileID.size());
  for (auto &IDChangePair : ChangesByFileID)
    Changes.push_back(std::move(IDChangePair.second));
  Consumer(llvm::MutableArrayRef<AtomicChange>(Changes));
}
