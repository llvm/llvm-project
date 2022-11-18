//===- DependencyScanningCASFilesystem.h - clang-scan-deps fs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGCASFILESYSTEM_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGCASFILESYSTEM_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningFilesystem.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/CAS/ThreadSafeFileSystem.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <mutex>

namespace llvm {
namespace cas {
class CachingOnDiskFileSystem;
} // namespace cas
} // namespace llvm

namespace clang {
namespace tooling {
namespace dependencies {

class DependencyScanningCASFilesystem : public llvm::cas::ThreadSafeFileSystem {
public:
  DependencyScanningCASFilesystem(
      IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> WorkerFS,
      llvm::cas::ActionCache &Cache);

  ~DependencyScanningCASFilesystem();

  // FIXME: Make a templated version of ProxyFileSystem with a configurable
  // base class.
  llvm::vfs::directory_iterator dir_begin(const Twine &Dir, std::error_code &EC) override {
    return FS->dir_begin(Dir, EC);
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return FS->getCurrentWorkingDirectory();
  }
  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    return FS->setCurrentWorkingDirectory(Path);
  }
  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) const override {
    return FS->getRealPath(Path, Output);
  }
  std::error_code isLocal(const Twine &Path, bool &Result) override {
    return FS->isLocal(Path, Result);
  }

  IntrusiveRefCntPtr<llvm::cas::ThreadSafeFileSystem>
  createThreadSafeProxyFS() override;

  llvm::ErrorOr<llvm::vfs::Status> status(const Twine &Path) override;
  bool exists(const Twine &Path) override;
  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const Twine &Path) override;

  /// \returns The scanned preprocessor directive tokens of the file that are
  /// used to speed up preprocessing, if available.
  Optional<ArrayRef<dependency_directives_scan::Directive>>
  getDirectiveTokens(const Twine &Path);

private:
  /// Check whether the file should be scanned for preprocessor directives.
  bool shouldScanForDirectives(StringRef Filename);

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;

  struct FileEntry {
    std::error_code EC; // If non-zero, caches a stat failure.
    Optional<StringRef> Buffer;
    SmallVector<dependency_directives_scan::Token, 64> DepTokens;
    SmallVector<dependency_directives_scan::Directive, 16> DepDirectives;
    llvm::vfs::Status Status;
    Optional<llvm::cas::ObjectRef> CASContents;
  };
  llvm::BumpPtrAllocator EntryAlloc;
  llvm::StringMap<FileEntry, llvm::BumpPtrAllocator &> Entries;

  struct LookupPathResult {
    const FileEntry *Entry = nullptr;

    // Only filled if the Entry is nullptr.
    llvm::ErrorOr<llvm::vfs::Status> Status;
  };
  void scanForDirectives(
      llvm::cas::ObjectRef InputDataID, StringRef Identifier,
      SmallVectorImpl<dependency_directives_scan::Token> &DepTokens,
      SmallVectorImpl<dependency_directives_scan::Directive> &DepDirectives);

  Expected<StringRef> getOriginal(llvm::cas::CASID InputDataID);

  LookupPathResult lookupPath(const Twine &Path);

  llvm::cas::CachingOnDiskFileSystem &getCachingFS();

  llvm::cas::ObjectStore &CAS;
  llvm::cas::ActionCache &Cache;
  Optional<llvm::cas::ObjectRef> ClangFullVersionID;
  Optional<llvm::cas::ObjectRef> DepDirectivesID;
  Optional<llvm::cas::ObjectRef> EmptyBlobID;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_DEPENDENCYSCANNINGCASFILESYSTEM_H
