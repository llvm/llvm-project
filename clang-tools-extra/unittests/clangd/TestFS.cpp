//===-- TestFS.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "TestFS.h"
#include "URI.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
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

/// unittest: is a scheme that refers to files relative to testRoot().
/// URI body is a path relative to testRoot() e.g. unittest:///x.h for
/// /clangd-test/x.h.
class TestScheme : public URIScheme {
public:
  static const char *Scheme;

  llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef /*Authority*/, llvm::StringRef Body,
                  llvm::StringRef HintPath) const override {
    assert(HintPath.startswith(testRoot()));
    if (!Body.consume_front("/"))
      return llvm::make_error<llvm::StringError>(
          "Body of an unittest: URI must start with '/'",
          llvm::inconvertibleErrorCode());
    llvm::SmallString<16> Path(Body.begin(), Body.end());
    llvm::sys::path::native(Path);
    return testPath(Path);
  }

  llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    llvm::StringRef Body = AbsolutePath;
    if (!Body.consume_front(testRoot()))
      return llvm::make_error<llvm::StringError>(
          AbsolutePath + "does not start with " + testRoot(),
          llvm::inconvertibleErrorCode());

    return URI(Scheme, /*Authority=*/"",
               llvm::sys::path::convert_to_slash(Body));
  }
};

const char *TestScheme::Scheme = "unittest";

static URISchemeRegistry::Add<TestScheme> X(TestScheme::Scheme, "Test schema");

volatile int UnittestSchemeAnchorSource = 0;

} // namespace clangd
} // namespace clang
