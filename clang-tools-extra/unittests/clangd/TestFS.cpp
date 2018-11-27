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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
using namespace llvm;

IntrusiveRefCntPtr<vfs::FileSystem>
buildTestFS(StringMap<std::string> const &Files,
            StringMap<time_t> const &Timestamps) {
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> MemFS(
      new vfs::InMemoryFileSystem);
  MemFS->setCurrentWorkingDirectory(testRoot());
  for (auto &FileAndContents : Files) {
    StringRef File = FileAndContents.first();
    MemFS->addFile(
        File, Timestamps.lookup(File),
        MemoryBuffer::getMemBufferCopy(FileAndContents.second, File));
  }
  return MemFS;
}

MockCompilationDatabase::MockCompilationDatabase(StringRef Directory,
                                                 StringRef RelPathPrefix)
    : ExtraClangFlags({"-ffreestanding"}), Directory(Directory),
      RelPathPrefix(RelPathPrefix) {
  // -ffreestanding avoids implicit stdc-predef.h.
}

Optional<tooling::CompileCommand>
MockCompilationDatabase::getCompileCommand(PathRef File,
                                           ProjectInfo *Project) const {
  if (ExtraClangFlags.empty())
    return None;

  auto FileName = sys::path::filename(File);

  // Build the compile command.
  auto CommandLine = ExtraClangFlags;
  CommandLine.insert(CommandLine.begin(), "clang");
  if (RelPathPrefix.empty()) {
    // Use the absolute path in the compile command.
    CommandLine.push_back(File);
  } else {
    // Build a relative path using RelPathPrefix.
    SmallString<32> RelativeFilePath(RelPathPrefix);
    sys::path::append(RelativeFilePath, FileName);
    CommandLine.push_back(RelativeFilePath.str());
  }

  if (Project)
    Project->SourceRoot = Directory;
  return {tooling::CompileCommand(
      Directory != StringRef() ? Directory : sys::path::parent_path(File),
      FileName, std::move(CommandLine), "")};
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

  Expected<std::string> getAbsolutePath(StringRef /*Authority*/, StringRef Body,
                                        StringRef HintPath) const override {
    assert(HintPath.startswith(testRoot()));
    if (!Body.consume_front("/"))
      return make_error<StringError>(
          "Body of an unittest: URI must start with '/'",
          inconvertibleErrorCode());
    SmallString<16> Path(Body.begin(), Body.end());
    sys::path::native(Path);
    return testPath(Path);
  }

  Expected<URI> uriFromAbsolutePath(StringRef AbsolutePath) const override {
    StringRef Body = AbsolutePath;
    if (!Body.consume_front(testRoot()))
      return make_error<StringError>(AbsolutePath + "does not start with " +
                                         testRoot(),
                                     inconvertibleErrorCode());

    return URI(Scheme, /*Authority=*/"", sys::path::convert_to_slash(Body));
  }
};

const char *TestScheme::Scheme = "unittest";

static URISchemeRegistry::Add<TestScheme> X(TestScheme::Scheme, "Test schema");

volatile int UnittestSchemeAnchorSource = 0;

} // namespace clangd
} // namespace clang
