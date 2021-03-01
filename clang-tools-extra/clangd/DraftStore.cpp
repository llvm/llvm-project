//===--- DraftStore.cpp - File contents container ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DraftStore.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"

namespace clang {
namespace clangd {

llvm::Optional<DraftStore::Draft> DraftStore::getDraft(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = Drafts.find(File);
  if (It == Drafts.end())
    return None;

  return It->second;
}

std::vector<Path> DraftStore::getActiveFiles() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  std::vector<Path> ResultVector;

  for (auto DraftIt = Drafts.begin(); DraftIt != Drafts.end(); DraftIt++)
    ResultVector.push_back(std::string(DraftIt->getKey()));

  return ResultVector;
}

static void increment(std::string &S) {
  // Ensure there is a numeric suffix.
  if (S.empty() || !llvm::isDigit(S.back())) {
    S.push_back('0');
    return;
  }
  // Increment the numeric suffix.
  auto I = S.rbegin(), E = S.rend();
  for (;;) {
    if (I == E || !llvm::isDigit(*I)) {
      // Reached start of numeric section, it was all 9s.
      S.insert(I.base(), '1');
      break;
    }
    if (*I != '9') {
      // Found a digit we can increment, we're done.
      ++*I;
      break;
    }
    *I = '0'; // and keep incrementing to the left.
  }
}

static void updateVersion(DraftStore::Draft &D,
                          llvm::StringRef SpecifiedVersion) {
  if (!SpecifiedVersion.empty()) {
    // We treat versions as opaque, but the protocol says they increase.
    if (SpecifiedVersion.compare_numeric(D.Version) <= 0)
      log("File version went from {0} to {1}", D.Version, SpecifiedVersion);
    D.Version = SpecifiedVersion.str();
  } else {
    // Note that if D was newly-created, this will bump D.Version from "" to 1.
    increment(D.Version);
  }
}

std::string DraftStore::addDraft(PathRef File, llvm::StringRef Version,
                                 llvm::StringRef Contents) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Draft &D = Drafts[File];
  updateVersion(D, Version);
  D.Contents = Contents.str();
  return D.Version;
}

void DraftStore::removeDraft(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts.erase(File);
}

} // namespace clangd
} // namespace clang
