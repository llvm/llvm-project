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
using namespace llvm;

IntrusiveRefCntPtr<vfs::FileSystem>
buildTestFS(StringMap<std::string> const &Files) {
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> MemFS(
      new vfs::InMemoryFileSystem);
  for (auto &FileAndContents : Files) {
    MemFS->addFile(FileAndContents.first(), time_t(),
                   MemoryBuffer::getMemBufferCopy(FileAndContents.second,
                                                  FileAndContents.first()));
  }
  return MemFS;
}

MockCompilationDatabase::MockCompilationDatabase(bool UseRelPaths)
    : ExtraClangFlags({"-ffreestanding"}), UseRelPaths(UseRelPaths) {
  // -ffreestanding avoids implicit stdc-predef.h.
}

Optional<tooling::CompileCommand>
MockCompilationDatabase::getCompileCommand(PathRef File) const {
  if (ExtraClangFlags.empty())
    return None;

  auto CommandLine = ExtraClangFlags;
  auto FileName = sys::path::filename(File);
  CommandLine.insert(CommandLine.begin(), "clang");
  CommandLine.insert(CommandLine.end(), UseRelPaths ? FileName : File);
  return {tooling::CompileCommand(sys::path::parent_path(File), FileName,
                                  std::move(CommandLine), "")};
}

const char *testRoot() {
#ifdef _WIN32
  return "C:\\clangd-test";
#else
  return "/clangd-test";
#endif
}

std::string testPath(PathRef File) {
  assert(sys::path::is_relative(File) && "FileName should be relative");

  SmallString<32> NativeFile = File;
  sys::path::native(NativeFile);
  SmallString<32> Path;
  sys::path::append(Path, testRoot(), NativeFile);
  return Path.str();
}

} // namespace clangd
} // namespace clang
