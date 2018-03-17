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

llvm::Optional<std::string> DraftStore::getDraft(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = Drafts.find(File);
  if (It == Drafts.end())
    return llvm::None;

  return It->second;
}

std::vector<Path> DraftStore::getActiveFiles() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  std::vector<Path> ResultVector;

  for (auto DraftIt = Drafts.begin(); DraftIt != Drafts.end(); DraftIt++)
    ResultVector.push_back(DraftIt->getKey());

  return ResultVector;
}

void DraftStore::updateDraft(PathRef File, StringRef Contents) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts[File] = Contents;
}

void DraftStore::removeDraft(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts.erase(File);
}
