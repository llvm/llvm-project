//===--- ThreadsafeFS.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/ThreadsafeFS.h"
#include "Logger.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <memory>

namespace clang {
namespace clangd {

namespace {
/// Always opens files in the underlying filesystem as "volatile", meaning they
/// won't be memory-mapped. Memory-mapping isn't desirable for clangd:
///   - edits to the underlying files change contents MemoryBuffers owned by
//      SourceManager, breaking its invariants and leading to crashes
///   - it locks files on windows, preventing edits
class VolatileFileSystem : public llvm::vfs::ProxyFileSystem {
public:
  explicit VolatileFileSystem(llvm::IntrusiveRefCntPtr<FileSystem> FS)
      : ProxyFileSystem(std::move(FS)) {}

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const llvm::Twine &InPath) override {
    llvm::SmallString<128> Path;
    InPath.toVector(Path);

    auto File = getUnderlyingFS().openFileForRead(Path);
    if (!File)
      return File;
    // Try to guess preamble files, they can be memory-mapped even on Windows as
    // clangd has exclusive access to those and nothing else should touch them.
    llvm::StringRef FileName = llvm::sys::path::filename(Path);
    if (FileName.startswith("preamble-") && FileName.endswith(".pch"))
      return File;
    return std::unique_ptr<VolatileFile>(new VolatileFile(std::move(*File)));
  }

private:
  class VolatileFile : public llvm::vfs::File {
  public:
    VolatileFile(std::unique_ptr<llvm::vfs::File> Wrapped)
        : Wrapped(std::move(Wrapped)) {
      assert(this->Wrapped);
    }

    virtual llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
    getBuffer(const llvm::Twine &Name, int64_t FileSize,
              bool RequiresNullTerminator, bool /*IsVolatile*/) override {
      return Wrapped->getBuffer(Name, FileSize, RequiresNullTerminator,
                                /*IsVolatile=*/true);
    }

    llvm::ErrorOr<llvm::vfs::Status> status() override {
      return Wrapped->status();
    }
    llvm::ErrorOr<std::string> getName() override { return Wrapped->getName(); }
    std::error_code close() override { return Wrapped->close(); }

  private:
    std::unique_ptr<File> Wrapped;
  };
};
} // namespace

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
ThreadsafeFS::view(PathRef CWD) const {
  auto FS = view(llvm::None);
  if (auto EC = FS->setCurrentWorkingDirectory(CWD))
    elog("VFS: failed to set CWD to {0}: {1}", CWD, EC.message());
  return FS;
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
RealThreadsafeFS::view(llvm::NoneType) const {
  // Avoid using memory-mapped files.
  // FIXME: Try to use a similar approach in Sema instead of relying on
  //        propagation of the 'isVolatile' flag through all layers.
  return new VolatileFileSystem(
      llvm::vfs::createPhysicalFileSystem().release());
}
} // namespace clangd
} // namespace clang
