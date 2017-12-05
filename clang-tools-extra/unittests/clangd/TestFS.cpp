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
namespace {

/// An implementation of vfs::FileSystem that only allows access to
/// files and folders inside a set of whitelisted directories.
///
/// FIXME(ibiryukov): should it also emulate access to parents of whitelisted
/// directories with only whitelisted contents?
class FilteredFileSystem : public vfs::FileSystem {
public:
  /// The paths inside \p WhitelistedDirs should be absolute
  FilteredFileSystem(std::vector<std::string> WhitelistedDirs,
                     IntrusiveRefCntPtr<vfs::FileSystem> InnerFS)
      : WhitelistedDirs(std::move(WhitelistedDirs)), InnerFS(InnerFS) {
    assert(std::all_of(WhitelistedDirs.begin(), WhitelistedDirs.end(),
                       [](const std::string &Path) -> bool {
                         return llvm::sys::path::is_absolute(Path);
                       }) &&
           "Not all WhitelistedDirs are absolute");
  }

  virtual llvm::ErrorOr<vfs::Status> status(const Twine &Path) {
    if (!isInsideWhitelistedDir(Path))
      return llvm::errc::no_such_file_or_directory;
    return InnerFS->status(Path);
  }

  virtual llvm::ErrorOr<std::unique_ptr<vfs::File>>
  openFileForRead(const Twine &Path) {
    if (!isInsideWhitelistedDir(Path))
      return llvm::errc::no_such_file_or_directory;
    return InnerFS->openFileForRead(Path);
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBufferForFile(const Twine &Name, int64_t FileSize = -1,
                   bool RequiresNullTerminator = true,
                   bool IsVolatile = false) {
    if (!isInsideWhitelistedDir(Name))
      return llvm::errc::no_such_file_or_directory;
    return InnerFS->getBufferForFile(Name, FileSize, RequiresNullTerminator,
                                     IsVolatile);
  }

  virtual vfs::directory_iterator dir_begin(const Twine &Dir,
                                            std::error_code &EC) {
    if (!isInsideWhitelistedDir(Dir)) {
      EC = llvm::errc::no_such_file_or_directory;
      return vfs::directory_iterator();
    }
    return InnerFS->dir_begin(Dir, EC);
  }

  virtual std::error_code setCurrentWorkingDirectory(const Twine &Path) {
    return InnerFS->setCurrentWorkingDirectory(Path);
  }

  virtual llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const {
    return InnerFS->getCurrentWorkingDirectory();
  }

  bool exists(const Twine &Path) {
    if (!isInsideWhitelistedDir(Path))
      return false;
    return InnerFS->exists(Path);
  }

  std::error_code makeAbsolute(SmallVectorImpl<char> &Path) const {
    return InnerFS->makeAbsolute(Path);
  }

private:
  bool isInsideWhitelistedDir(const Twine &InputPath) const {
    SmallString<128> Path;
    InputPath.toVector(Path);

    if (makeAbsolute(Path))
      return false;

    for (const auto &Dir : WhitelistedDirs) {
      if (Path.startswith(Dir))
        return true;
    }
    return false;
  }

  std::vector<std::string> WhitelistedDirs;
  IntrusiveRefCntPtr<vfs::FileSystem> InnerFS;
};

/// Create a vfs::FileSystem that has access only to temporary directories
/// (obtained by calling system_temp_directory).
IntrusiveRefCntPtr<vfs::FileSystem> getTempOnlyFS() {
  llvm::SmallString<128> TmpDir1;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/false, TmpDir1);
  llvm::SmallString<128> TmpDir2;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, TmpDir2);

  std::vector<std::string> TmpDirs;
  TmpDirs.push_back(TmpDir1.str());
  if (TmpDir1 != TmpDir2)
    TmpDirs.push_back(TmpDir2.str());
  return new FilteredFileSystem(std::move(TmpDirs), vfs::getRealFileSystem());
}

} // namespace

IntrusiveRefCntPtr<vfs::FileSystem>
buildTestFS(llvm::StringMap<std::string> const &Files) {
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> MemFS(
      new vfs::InMemoryFileSystem);
  for (auto &FileAndContents : Files)
    MemFS->addFile(FileAndContents.first(), time_t(),
                   llvm::MemoryBuffer::getMemBuffer(FileAndContents.second,
                                                    FileAndContents.first()));

  auto OverlayFS = IntrusiveRefCntPtr<vfs::OverlayFileSystem>(
      new vfs::OverlayFileSystem(getTempOnlyFS()));
  OverlayFS->pushOverlay(std::move(MemFS));
  return OverlayFS;
}

Tagged<IntrusiveRefCntPtr<vfs::FileSystem>>
MockFSProvider::getTaggedFileSystem(PathRef File) {
  if (ExpectedFile) {
    EXPECT_EQ(*ExpectedFile, File);
  }

  auto FS = buildTestFS(Files);
  return make_tagged(FS, Tag);
}

MockCompilationDatabase::MockCompilationDatabase()
    : ExtraClangFlags({"-ffreestanding"}) {} // Avoid implicit stdc-predef.h.

llvm::Optional<tooling::CompileCommand>
MockCompilationDatabase::getCompileCommand(PathRef File) const {
  if (ExtraClangFlags.empty())
    return llvm::None;

  auto CommandLine = ExtraClangFlags;
  CommandLine.insert(CommandLine.begin(), "clang");
  CommandLine.insert(CommandLine.end(), File.str());
  return {tooling::CompileCommand(llvm::sys::path::parent_path(File),
                                  llvm::sys::path::filename(File),
                                  std::move(CommandLine), "")};
}

static const char *getVirtualTestRoot() {
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
