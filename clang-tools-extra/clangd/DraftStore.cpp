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

static void updateVersion(DraftStore::Draft &D,
                          llvm::Optional<int64_t> Version) {
  if (Version) {
    // We treat versions as opaque, but the protocol says they increase.
    if (*Version <= D.Version)
      log("File version went from {0} to {1}", D.Version, Version);
    D.Version = *Version;
  } else {
    // Note that if D was newly-created, this will bump D.Version from -1 to 0.
    ++D.Version;
  }
}

int64_t DraftStore::addDraft(PathRef File, llvm::Optional<int64_t> Version,
                         llvm::StringRef Contents) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Draft &D = Drafts[File];
  updateVersion(D, Version);
  D.Contents = Contents.str();
  return D.Version;
}

llvm::Expected<DraftStore::Draft> DraftStore::updateDraft(
    PathRef File, llvm::Optional<int64_t> Version,
    llvm::ArrayRef<TextDocumentContentChangeEvent> Changes) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto EntryIt = Drafts.find(File);
  if (EntryIt == Drafts.end()) {
    return error(llvm::errc::invalid_argument,
                 "Trying to do incremental update on non-added document: {0}",
                 File);
  }
  Draft &D = EntryIt->second;
  std::string Contents = EntryIt->second.Contents;

  for (const TextDocumentContentChangeEvent &Change : Changes) {
    if (!Change.range) {
      Contents = Change.text;
      continue;
    }

    const Position &Start = Change.range->start;
    llvm::Expected<size_t> StartIndex =
        positionToOffset(Contents, Start, false);
    if (!StartIndex)
      return StartIndex.takeError();

    const Position &End = Change.range->end;
    llvm::Expected<size_t> EndIndex = positionToOffset(Contents, End, false);
    if (!EndIndex)
      return EndIndex.takeError();

    if (*EndIndex < *StartIndex)
      return error(llvm::errc::invalid_argument,
                   "Range's end position ({0}) is before start position ({1})",
                   End, Start);

    // Since the range length between two LSP positions is dependent on the
    // contents of the buffer we compute the range length between the start and
    // end position ourselves and compare it to the range length of the LSP
    // message to verify the buffers of the client and server are in sync.

    // EndIndex and StartIndex are in bytes, but Change.rangeLength is in UTF-16
    // code units.
    ssize_t ComputedRangeLength =
        lspLength(Contents.substr(*StartIndex, *EndIndex - *StartIndex));

    if (Change.rangeLength && ComputedRangeLength != *Change.rangeLength)
      return error(llvm::errc::invalid_argument,
                   "Change's rangeLength ({0}) doesn't match the "
                   "computed range length ({1}).",
                   *Change.rangeLength, ComputedRangeLength);

    std::string NewContents;
    NewContents.reserve(*StartIndex + Change.text.length() +
                        (Contents.length() - *EndIndex));

    NewContents = Contents.substr(0, *StartIndex);
    NewContents += Change.text;
    NewContents += Contents.substr(*EndIndex);

    Contents = std::move(NewContents);
  }

  updateVersion(D, Version);
  D.Contents = std::move(Contents);
  return D;
}

void DraftStore::removeDraft(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts.erase(File);
}

} // namespace clangd
} // namespace clang
