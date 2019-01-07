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

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
buildTestFS(llvm::StringMap<std::string> const &Files,
            llvm::StringMap<time_t> const &Timestamps) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> MemFS(
      new llvm::vfs::InMemoryFileSystem);
  MemFS->setCurrentWorkingDirectory(testRoot());
  for (auto &FileAndContents : Files) {
    llvm::StringRef File = FileAndContents.first();
    MemFS->addFile(
        File, Timestamps.lookup(File),
        llvm::MemoryBuffer::getMemBufferCopy(FileAndContents.second, File));
  }
  return MemFS;
}

MockCompilationDatabase::MockCompilationDatabase(llvm::StringRef Directory,
                                                 llvm::StringRef RelPathPrefix)
    : ExtraClangFlags({"-ffreestanding"}), Directory(Directory),
      RelPathPrefix(RelPathPrefix) {
  // -ffreestanding avoids implicit stdc-predef.h.
}

llvm::Optional<tooling::CompileCommand>
MockCompilationDatabase::getCompileCommand(PathRef File,
                                           ProjectInfo *Project) const {
  if (ExtraClangFlags.empty())
    return None;

  auto FileName = llvm::sys::path::filename(File);

  // Build the compile command.
  auto CommandLine = ExtraClangFlags;
  CommandLine.insert(CommandLine.begin(), "clang");
  if (RelPathPrefix.empty()) {
    // Use the absolute path in the compile command.
    CommandLine.push_back(File);
  } else {
    // Build a relative path using RelPathPrefix.
    llvm::SmallString<32> RelativeFilePath(RelPathPrefix);
    llvm::sys::path::append(RelativeFilePath, FileName);
    CommandLine.push_back(RelativeFilePath.str());
  }

  if (Project)
    Project->SourceRoot = Directory;
  return {tooling::CompileCommand(Directory != llvm::StringRef()
                                      ? Directory
                                      : llvm::sys::path::parent_path(File),
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
  assert(llvm::sys::path::is_relative(File) && "FileName should be relative");

  llvm::SmallString<32> NativeFile = File;
  llvm::sys::path::native(NativeFile);
  llvm::SmallString<32> Path;
  llvm::sys::path::append(Path, testRoot(), NativeFile);
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
    if (!HintPath.startswith(testRoot()))
      return llvm::make_error<llvm::StringError>(
          "Hint path doesn't start with test root: " + HintPath,
          llvm::inconvertibleErrorCode());
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
