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
  Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
  getTaggedFileSystem(PathRef File) override;

  llvm::Optional<SmallString<32>> ExpectedFile;
  llvm::StringMap<std::string> Files;
  VFSTag Tag = VFSTag();
};

// A Compilation database that returns a fixed set of compile flags.
class MockCompilationDatabase : public GlobalCompilationDatabase {
public:
  MockCompilationDatabase(bool AddFreestandingFlag) {
    // We have to add -ffreestanding to VFS-specific tests to avoid errors on
    // implicit includes of stdc-predef.h.
    if (AddFreestandingFlag)
      ExtraClangFlags.push_back("-ffreestanding");
  }

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override;

  std::vector<std::string> ExtraClangFlags;
};

// Returns a suitable absolute path for this OS.
llvm::SmallString<32> getVirtualTestFilePath(PathRef File);

} // namespace clangd
} // namespace clang
