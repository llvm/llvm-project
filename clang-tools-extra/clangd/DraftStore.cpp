//===--- DraftStore.cpp - File contents container ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DraftStore.h"
#include "support/Logger.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <chrono>
#include <memory>
#include <optional>
#include <system_error>

namespace clang {
namespace clangd {

std::optional<DraftStore::Draft> DraftStore::getDraft(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = Drafts.find(File);
  if (It == Drafts.end())
    return std::nullopt;

  return It->second.D;
}

std::vector<Path> DraftStore::getActiveFiles() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  std::vector<Path> ResultVector;

  for (const auto &Draft : Drafts)
    ResultVector.push_back(Draft.first);

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
using PathStyle = llvm::sys::path::Style;

PathStyle pathStyle(llvm::StringRef Path) {
  return hasWindowsDrive(Path) ? PathStyle::windows : PathStyle::native;
}

/// A read-only MemoryBuffer that keeps the draft contents alive.
class SharedStringBuffer : public llvm::MemoryBuffer {
  std::shared_ptr<const std::string> Contents;
  std::string Name;

public:
  SharedStringBuffer(std::shared_ptr<const std::string> Contents,
                     llvm::StringRef Name)
      : Contents(std::move(Contents)), Name(Name) {
    assert(this->Contents && "draft contents must be present");
    init(this->Contents->c_str(),
         this->Contents->c_str() + this->Contents->size(),
         /*RequiresNullTerminator=*/true);
  }

  BufferKind getBufferKind() const override {
    return MemoryBuffer::MemoryBuffer_Malloc;
  }

  llvm::StringRef getBufferIdentifier() const override { return Name; }
};

struct DraftNode {
  struct Child {
    std::string Name;
    llvm::sys::fs::file_type Type;
  };

  enum Kind { File, Directory } K = Directory;
  llvm::sys::fs::UniqueID ID;
  std::shared_ptr<const std::string> Contents;
  std::time_t MTime = 0;
  llvm::SmallVector<Child, 2> Children;

  explicit DraftNode(
      llvm::sys::fs::UniqueID ID = llvm::vfs::getNextVirtualUniqueID())
      : ID(ID) {}
};

class DraftFile : public llvm::vfs::File {
  llvm::vfs::Status Stat;
  std::shared_ptr<const std::string> Contents;

public:
  DraftFile(llvm::vfs::Status S, std::shared_ptr<const std::string> C)
      : Stat(std::move(S)), Contents(std::move(C)) {}

  llvm::ErrorOr<llvm::vfs::Status> status() override { return Stat; }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t /*FileSize*/,
            bool RequiresNullTerminator, bool /*IsVolatile*/) override {
    (void)RequiresNullTerminator;
    return std::make_unique<SharedStringBuffer>(Contents, Name.str());
  }

  std::error_code close() override { return {}; }
};

class DraftDirIterator : public llvm::vfs::detail::DirIterImpl {
  llvm::ArrayRef<DraftNode::Child> Children;
  std::string RequestedDir;
  PathStyle Style;
  size_t Index = 0;

  void setCurrentEntry() {
    if (Index == Children.size()) {
      CurrentEntry = {};
      return;
    }
    llvm::SmallString<256> Path(RequestedDir);
    llvm::sys::path::append(Path, Style, Children[Index].Name);
    CurrentEntry =
        llvm::vfs::directory_entry(Path.str().str(), Children[Index].Type);
  }

public:
  DraftDirIterator(llvm::ArrayRef<DraftNode::Child> Children,
                   std::string RequestedDir, PathStyle Style)
      : Children(Children), RequestedDir(std::move(RequestedDir)),
        Style(Style) {
    setCurrentEntry();
  }

  std::error_code increment() override {
    ++Index;
    setCurrentEntry();
    return {};
  }
};

/// Overlay whose lookups use Path identity (drive letter, slashes) rather than
/// dumping first-inserted spellings into a case-sensitive InMemoryFileSystem.
class DraftsFileSystem : public llvm::vfs::FileSystem {
  using NodeMap = PathMap<DraftNode>;
  NodeMap Nodes;
  std::string CWD;

  static llvm::vfs::Status makeStatus(llvm::StringRef Requested,
                                      const DraftNode &N) {
    const bool IsFile = N.K == DraftNode::File;
    return llvm::vfs::Status(Requested, N.ID,
                             std::chrono::system_clock::from_time_t(N.MTime),
                             /*User=*/0,
                             /*Group=*/0, IsFile ? N.Contents->size() : 0,
                             IsFile ? llvm::sys::fs::file_type::regular_file
                                    : llvm::sys::fs::file_type::directory_file,
                             llvm::sys::fs::all_all);
  }

  PathRef resolve(const llvm::Twine &Requested,
                  llvm::SmallString<256> &Storage) const {
    Requested.toVector(Storage);
    PathStyle Style = pathStyle(Storage);
    if (llvm::sys::path::is_relative(Storage, Style) && !CWD.empty()) {
      llvm::sys::path::make_absolute(CWD, Storage);
      Style = pathStyle(Storage);
    }
    llvm::sys::path::remove_dots(Storage, /*remove_dot_dot=*/true, Style);
    return Storage;
  }

  NodeMap::const_iterator lookup(const llvm::Twine &Path,
                                 llvm::SmallString<256> &Storage) const {
    return Nodes.find(resolve(Path, Storage));
  }

public:
  explicit DraftsFileSystem(PathMap<DraftNode> Nodes)
      : Nodes(std::move(Nodes)) {}

  llvm::ErrorOr<llvm::vfs::Status> status(const llvm::Twine &Path) override {
    llvm::SmallString<256> Storage;
    auto It = lookup(Path, Storage);
    if (It != Nodes.end())
      return makeStatus(Path.str(), It->second);
    return llvm::errc::no_such_file_or_directory;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const llvm::Twine &Path) override {
    llvm::SmallString<256> Storage;
    auto It = lookup(Path, Storage);
    if (It == Nodes.end())
      return llvm::errc::no_such_file_or_directory;
    if (It->second.K != DraftNode::File)
      return llvm::errc::invalid_argument;
    return std::unique_ptr<llvm::vfs::File>(std::make_unique<DraftFile>(
        makeStatus(Path.str(), It->second), It->second.Contents));
  }

  llvm::vfs::directory_iterator dir_begin(const llvm::Twine &Path,
                                          std::error_code &EC) override {
    llvm::SmallString<256> Storage;
    auto It = lookup(Path, Storage);
    if (It == Nodes.end()) {
      EC = llvm::errc::no_such_file_or_directory;
      return {};
    }
    if (It->second.K != DraftNode::Directory) {
      EC = llvm::errc::not_a_directory;
      return {};
    }
    EC = {};
    return llvm::vfs::directory_iterator(std::make_shared<DraftDirIterator>(
        It->second.Children, Path.str(), pathStyle(Storage)));
  }

  std::error_code setCurrentWorkingDirectory(const llvm::Twine &Path) override {
    llvm::SmallString<256> Storage;
    CWD = resolve(Path, Storage).raw().str();
    return {};
  }

  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    if (CWD.empty())
      return llvm::errc::no_such_file_or_directory;
    return CWD;
  }

  std::error_code getRealPath(const llvm::Twine &Path,
                              llvm::SmallVectorImpl<char> &Output) override {
    llvm::SmallString<256> Storage;
    auto It = lookup(Path, Storage);
    if (It == Nodes.end())
      return llvm::errc::no_such_file_or_directory;
    Output.clear();
    Output.append(It->first.raw().begin(), It->first.raw().end());
    return {};
  }

  std::error_code isLocal(const llvm::Twine &Path, bool &Result) override {
    llvm::SmallString<256> Storage;
    if (lookup(Path, Storage) == Nodes.end())
      return llvm::errc::no_such_file_or_directory;
    Result = false;
    return {};
  }
};
} // namespace

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> DraftStore::asVFS() const {
  PathMap<DraftNode> Snapshot;
  {
    std::lock_guard<std::mutex> Guard(Mutex);
    for (const auto &Draft : Drafts) {
      llvm::SmallString<256> Canonical(Draft.first.raw());
      PathStyle Style = pathStyle(Canonical);
      llvm::sys::path::remove_dots(Canonical, /*remove_dot_dot=*/true, Style);
      auto FileIt =
          Snapshot.try_emplace(PathRef(Canonical), Draft.second.ID).first;
      FileIt->second.K = DraftNode::File;
      FileIt->second.ID = Draft.second.ID;
      FileIt->second.Contents = Draft.second.D.Contents;
      FileIt->second.MTime = Draft.second.MTime;

      PathRef Child = Canonical;
      while (!llvm::sys::path::relative_path(Child.raw(), Style).empty()) {
        llvm::StringRef Parent =
            llvm::sys::path::parent_path(Child.raw(), Style);
        if (Parent.empty() || Parent == Child.raw())
          break;
        Snapshot.try_emplace(PathRef(Parent));
        Child = Parent;
      }
    }
  }

  // Populate directory adjacency only after all nodes have been inserted, so
  // DenseMap rehashing cannot invalidate any node references.
  for (const auto &Entry : Snapshot) {
    PathStyle Style = pathStyle(Entry.first.raw());
    if (llvm::sys::path::relative_path(Entry.first.raw(), Style).empty())
      continue;
    llvm::StringRef Parent =
        llvm::sys::path::parent_path(Entry.first.raw(), Style);
    if (Parent.empty())
      continue;
    auto ParentIt = Snapshot.find(Parent);
    if (ParentIt == Snapshot.end() ||
        ParentIt->second.K != DraftNode::Directory)
      continue;
    ParentIt->second.Children.push_back(
        {llvm::sys::path::filename(Entry.first.raw(), Style).str(),
         Entry.second.K == DraftNode::File
             ? llvm::sys::fs::file_type::regular_file
             : llvm::sys::fs::file_type::directory_file});
  }
  return llvm::makeIntrusiveRefCnt<DraftsFileSystem>(std::move(Snapshot));
}
} // namespace clangd
} // namespace clang
