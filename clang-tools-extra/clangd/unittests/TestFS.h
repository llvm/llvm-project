//===-- TestFS.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Allows setting up fake filesystem environments for tests.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTFS_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTFS_H
#include "ClangdServer.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace clang {
namespace clangd {

// Builds a VFS that provides access to the provided files, plus temporary
// directories.
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
buildTestFS(llvm::StringMap<std::string> const &Files,
            llvm::StringMap<time_t> const &Timestamps = {});

// A VFS provider that returns TestFSes containing a provided set of files.
class MockFSProvider : public FileSystemProvider {
public:
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> getFileSystem() const override {
    return buildTestFS(Files);
  }

  // If relative paths are used, they are resolved with testPath().
  llvm::StringMap<std::string> Files;
};

// A Compilation database that returns a fixed set of compile flags.
class MockCompilationDatabase : public GlobalCompilationDatabase {
public:
  /// If \p Directory is not empty, use that as the Directory field of the
  /// CompileCommand, and as project SourceRoot.
  ///
  /// If \p RelPathPrefix is not empty, use that as a prefix in front of the
  /// source file name, instead of using an absolute path.
  MockCompilationDatabase(StringRef Directory = StringRef(),
                          StringRef RelPathPrefix = StringRef());

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File, ProjectInfo * = nullptr) const override;

  std::vector<std::string> ExtraClangFlags;

private:
  StringRef Directory;
  StringRef RelPathPrefix;
};

// Returns an absolute (fake) test directory for this OS.
const char *testRoot();

// Returns a suitable absolute path for this OS.
std::string testPath(PathRef File);

// unittest: is a scheme that refers to files relative to testRoot()
// This anchor is used to force the linker to link in the generated object file
// and thus register unittest: URI scheme plugin.
extern volatile int UnittestSchemeAnchorSource;

} // namespace clangd
} // namespace clang
#endif
