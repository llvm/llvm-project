//===--- DraftStore.cpp - File contents container ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DraftStore.h"
#include "SourceCode.h"
#include "llvm/Support/Errc.h"

using namespace llvm;
namespace clang {
namespace clangd {

Optional<std::string> DraftStore::getDraft(PathRef File) const {
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
    ResultVector.push_back(DraftIt->getKey());

  return ResultVector;
}

void DraftStore::addDraft(PathRef File, StringRef Contents) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts[File] = Contents;
}

Expected<std::string>
DraftStore::updateDraft(PathRef File,
                        ArrayRef<TextDocumentContentChangeEvent> Changes) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto EntryIt = Drafts.find(File);
  if (EntryIt == Drafts.end()) {
    return make_error<StringError>(
        "Trying to do incremental update on non-added document: " + File,
        llvm::errc::invalid_argument);
  }

  std::string Contents = EntryIt->second;

  for (const TextDocumentContentChangeEvent &Change : Changes) {
    if (!Change.range) {
      Contents = Change.text;
      continue;
    }

    const Position &Start = Change.range->start;
    Expected<size_t> StartIndex = positionToOffset(Contents, Start, false);
    if (!StartIndex)
      return StartIndex.takeError();

    const Position &End = Change.range->end;
    Expected<size_t> EndIndex = positionToOffset(Contents, End, false);
    if (!EndIndex)
      return EndIndex.takeError();

    if (*EndIndex < *StartIndex)
      return make_error<StringError>(
          formatv("Range's end position ({0}) is before start position ({1})",
                  End, Start),
          llvm::errc::invalid_argument);

    // Since the range length between two LSP positions is dependent on the
    // contents of the buffer we compute the range length between the start and
    // end position ourselves and compare it to the range length of the LSP
    // message to verify the buffers of the client and server are in sync.

    // EndIndex and StartIndex are in bytes, but Change.rangeLength is in UTF-16
    // code units.
    ssize_t ComputedRangeLength =
        lspLength(Contents.substr(*StartIndex, *EndIndex - *StartIndex));

    if (Change.rangeLength && ComputedRangeLength != *Change.rangeLength)
      return make_error<StringError>(
          formatv("Change's rangeLength ({0}) doesn't match the "
                  "computed range length ({1}).",
                  *Change.rangeLength, *EndIndex - *StartIndex),
          llvm::errc::invalid_argument);

    std::string NewContents;
    NewContents.reserve(*StartIndex + Change.text.length() +
                        (Contents.length() - *EndIndex));

    NewContents = Contents.substr(0, *StartIndex);
    NewContents += Change.text;
    NewContents += Contents.substr(*EndIndex);

    Contents = std::move(NewContents);
  }

  EntryIt->second = Contents;
  return Contents;
}

void DraftStore::removeDraft(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts.erase(File);
}

} // namespace clangd
} // namespace clang
