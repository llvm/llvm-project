//===-- TestFS.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Allows setting up fake filesystem environments for tests.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTFS_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_TESTFS_H
#include "ClangdServer.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {

// Builds a VFS that provides access to the provided files, plus temporary
// directories.
llvm::IntrusiveRefCntPtr<vfs::FileSystem>
buildTestFS(llvm::StringMap<std::string> const &Files);

// A VFS provider that returns TestFSes containing a provided set of files.
class MockFSProvider : public FileSystemProvider {
public:
  IntrusiveRefCntPtr<vfs::FileSystem> getFileSystem() override {
    return buildTestFS(Files);
  }

  // If relative paths are used, they are resolved with testPath().
  llvm::StringMap<std::string> Files;
};

// A Compilation database that returns a fixed set of compile flags.
class MockCompilationDatabase : public GlobalCompilationDatabase {
public:
  /// When \p UseRelPaths is true, uses relative paths in compile commands.
  /// When \p UseRelPaths is false, uses absoulte paths.
  MockCompilationDatabase(bool UseRelPaths = false);

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override;

  std::vector<std::string> ExtraClangFlags;
  const bool UseRelPaths;
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
