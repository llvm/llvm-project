//===-- TestFS.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "TestFS.h"
#include "llvm/Support/Errc.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
IntrusiveRefCntPtr<vfs::FileSystem>
buildTestFS(llvm::StringMap<std::string> const &Files) {
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> MemFS(
      new vfs::InMemoryFileSystem);
  for (auto &FileAndContents : Files)
    MemFS->addFile(FileAndContents.first(), time_t(),
                   llvm::MemoryBuffer::getMemBuffer(FileAndContents.second,
                                                    FileAndContents.first()));
  return MemFS;
}

Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
MockFSProvider::getTaggedFileSystem(PathRef File) {
  if (ExpectedFile) {
    EXPECT_EQ(*ExpectedFile, File);
  }

  auto FS = buildTestFS(Files);
  return make_tagged(FS, Tag);
}

MockCompilationDatabase::MockCompilationDatabase(bool UseRelPaths)
    : ExtraClangFlags({"-ffreestanding"}), UseRelPaths(UseRelPaths) {
  // -ffreestanding avoids implicit stdc-predef.h.
}

llvm::Optional<tooling::CompileCommand>
MockCompilationDatabase::getCompileCommand(PathRef File) const {
  if (ExtraClangFlags.empty())
    return llvm::None;

  auto CommandLine = ExtraClangFlags;
  auto FileName = llvm::sys::path::filename(File);
  CommandLine.insert(CommandLine.begin(), "clang");
  CommandLine.insert(CommandLine.end(), UseRelPaths ? FileName : File);
  return {tooling::CompileCommand(llvm::sys::path::parent_path(File), FileName,
                                  std::move(CommandLine), "")};
}

const char *getVirtualTestRoot() {
#ifdef LLVM_ON_WIN32
  return "C:\\clangd-test";
#else
  return "/clangd-test";
#endif
}

llvm::SmallString<32> getVirtualTestFilePath(PathRef File) {
  assert(llvm::sys::path::is_relative(File) && "FileName should be relative");

  llvm::SmallString<32> Path;
  llvm::sys::path::append(Path, getVirtualTestRoot(), File);
  return Path;
}

} // namespace clangd
} // namespace clang
