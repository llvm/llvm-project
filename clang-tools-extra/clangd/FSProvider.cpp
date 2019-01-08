//===--- FSProvider.cpp - VFS provider for ClangdServer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FSProvider.h"
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
/// won't be memory-mapped. This avoid locking the files on Windows.
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
    // clangd has exclusive access to those.
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
clang::clangd::RealFileSystemProvider::getFileSystem() const {
// Avoid using memory-mapped files on Windows, they cause file locking issues.
// FIXME: Try to use a similar approach in Sema instead of relying on
//        propagation of the 'isVolatile' flag through all layers.
#ifdef _WIN32
  return new VolatileFileSystem(llvm::vfs::getRealFileSystem());
#else
  return llvm::vfs::getRealFileSystem();
#endif
}
} // namespace clangd
} // namespace clang
