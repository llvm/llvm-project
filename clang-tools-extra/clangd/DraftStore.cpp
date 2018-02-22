//===--- DraftStore.cpp - File contents container ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DraftStore.h"

using namespace clang;
using namespace clang::clangd;

VersionedDraft DraftStore::getDraft(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = Drafts.find(File);
  if (It == Drafts.end())
    return {0, llvm::None};
  return It->second;
}

std::vector<Path> DraftStore::getActiveFiles() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  std::vector<Path> ResultVector;

  for (auto DraftIt = Drafts.begin(); DraftIt != Drafts.end(); DraftIt++)
    if (DraftIt->second.Draft)
      ResultVector.push_back(DraftIt->getKey());

  return ResultVector;
}

DocVersion DraftStore::getVersion(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = Drafts.find(File);
  if (It == Drafts.end())
    return 0;
  return It->second.Version;
}

DocVersion DraftStore::updateDraft(PathRef File, StringRef Contents) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto &Entry = Drafts[File];
  DocVersion NewVersion = ++Entry.Version;
  Entry.Draft = Contents;
  return NewVersion;
}

DocVersion DraftStore::removeDraft(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto &Entry = Drafts[File];
  DocVersion NewVersion = ++Entry.Version;
  Entry.Draft = llvm::None;
  return NewVersion;
}
