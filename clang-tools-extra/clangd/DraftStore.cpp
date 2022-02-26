//===--- DraftStore.cpp - File contents container ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DraftStore.h"
#include "support/Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>

namespace clang {
namespace clangd {

llvm::Optional<DraftStore::Draft> DraftStore::getDraft(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = Drafts.find(File);
  if (It == Drafts.end())
    return None;

  return It->second.D;
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

  auto &D = Drafts[File];
  updateVersion(D.D, Version);
  std::time(&D.MTime);
  D.D.Contents = std::make_shared<std::string>(Contents);
  return D.D.Version;
}

void DraftStore::removeDraft(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  Drafts.erase(File);
}

namespace {

/// A read only MemoryBuffer shares ownership of a ref counted string. The
/// shared string object must not be modified while an owned by this buffer.
class SharedStringBuffer : public llvm::MemoryBuffer {
  const std::shared_ptr<const std::string> BufferContents;
  const std::string Name;

public:
  BufferKind getBufferKind() const override {
    return MemoryBuffer::MemoryBuffer_Malloc;
  }

  StringRef getBufferIdentifier() const override { return Name; }

  SharedStringBuffer(std::shared_ptr<const std::string> Data, StringRef Name)
      : BufferContents(std::move(Data)), Name(Name) {
    assert(BufferContents && "Can't create from empty shared_ptr");
    MemoryBuffer::init(BufferContents->c_str(),
                       BufferContents->c_str() + BufferContents->size(),
                       /*RequiresNullTerminator=*/true);
  }
};
} // namespace

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> DraftStore::asVFS() const {
  auto MemFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  std::lock_guard<std::mutex> Guard(Mutex);
  for (const auto &Draft : Drafts)
    MemFS->addFile(Draft.getKey(), Draft.getValue().MTime,
                   std::make_unique<SharedStringBuffer>(
                       Draft.getValue().D.Contents, Draft.getKey()));
  return MemFS;
}
} // namespace clangd
} // namespace clang
