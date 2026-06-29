//===- unittests/Frontend/CachedDiagnosticsTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/Frontend/CachedDiagnostics.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrefixMapper.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace clang;
using namespace clang::cas;

namespace {

constexpr llvm::StringLiteral TestPath = "/test.c";
constexpr llvm::StringLiteral TestContents = "int x;\n";

static llvm::IntrusiveRefCntPtr<FileManager>
makeInMemoryFileManager(StringRef Path, StringRef Contents) {
  auto VFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  VFS->addFile(Path, /*ModificationTime=*/0,
               llvm::MemoryBuffer::getMemBufferCopy(Contents, Path));
  return llvm::makeIntrusiveRefCnt<FileManager>(FileSystemOptions(),
                                                std::move(VFS));
}

/// A VFS proxy that can inject errors on demand.
class FailableProxyFS : public llvm::vfs::ProxyFileSystem {
public:
  FailableProxyFS(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> Underlying)
      : ProxyFileSystem(std::move(Underlying)) {}

  bool FailOpens = false;

  llvm::ErrorOr<std::unique_ptr<llvm::vfs::File>>
  openFileForRead(const llvm::Twine &Path) override {
    if (FailOpens)
      return std::make_error_code(std::errc::no_such_file_or_directory);
    return ProxyFileSystem::openFileForRead(Path);
  }
};

/// Capture and serialize one warning emitted at the start of \p Path.
static std::string captureOneWarning(FileManager &FileMgr, StringRef Path) {
  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags(new DiagnosticsEngine(
      DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()));
  SourceManager SrcMgr(*Diags, FileMgr);
  Diags->setSourceManager(&SrcMgr);

  llvm::PrefixMapper Mapper;
  CachingDiagnosticsProcessor Processor(std::move(Mapper), FileMgr);
  Processor.insertDiagConsumer(*Diags);

  auto FE = FileMgr.getFileRef(Path);
  EXPECT_TRUE((bool)FE);
  FileID FID = SrcMgr.createFileID(*FE, SourceLocation(), SrcMgr::C_User);
  SourceLocation Loc = SrcMgr.getLocForStartOfFile(FID);

  unsigned DiagID =
      Diags->getCustomDiagID(DiagnosticsEngine::Warning, "test diagnostic");
  Diags->Report(Loc, DiagID);

  auto Serialized = Processor.serializeEmittedDiagnostics();
  EXPECT_THAT_EXPECTED(Serialized, llvm::Succeeded());
  Processor.removeDiagConsumer(*Diags);
  if (!Serialized || !*Serialized)
    return {};
  return **Serialized;
}

TEST(CachedDiagnosticsTest, ReplayRoundTrip) {
  auto FileMgr = makeInMemoryFileManager(TestPath, TestContents);
  std::string Serialized = captureOneWarning(*FileMgr, TestPath);
  ASSERT_FALSE(Serialized.empty());

  llvm::PrefixMapper Mapper;
  CachingDiagnosticsProcessor Replay(std::move(Mapper), *FileMgr);
  TextDiagnosticBuffer Consumer;
  EXPECT_THAT_ERROR(Replay.replayCachedDiagnostics(Serialized, Consumer),
                    llvm::Succeeded());
  EXPECT_EQ(Consumer.getNumWarnings(), 1u);
}

TEST(CachedDiagnosticsTest, ReplayWithMissingFileReturnsError) {
  // Capture diagnostics referencing a real file (only the path is recorded).
  auto FileMgr = makeInMemoryFileManager(TestPath, TestContents);
  std::string Serialized = captureOneWarning(*FileMgr, TestPath);
  ASSERT_FALSE(Serialized.empty());

  // Replay against a fresh FileManager whose VFS does not contain the file.
  auto EmptyVFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  auto EmptyFileMgr = llvm::makeIntrusiveRefCnt<FileManager>(
      FileSystemOptions(), std::move(EmptyVFS));

  llvm::PrefixMapper Mapper;
  CachingDiagnosticsProcessor Replay(std::move(Mapper), *EmptyFileMgr);
  TextDiagnosticBuffer Consumer;
  EXPECT_THAT_ERROR(Replay.replayCachedDiagnostics(Serialized, Consumer),
                    llvm::Failed());
}

TEST(CachedDiagnosticsTest, CaptureFailureAbortsSerialization) {
  // We jump through hoops here to create a situation that we don't expect to
  // occur normally, which is for a file buffer to be unavailable during diag
  // serialization. One way this can happen in real compiles is a diagnostic
  // located in a file loaded via pch/pcm but not yet opened. Here we do it by
  // manipulating the VFS to inject an error.

  auto InMem = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMem->addFile(TestPath, /*ModificationTime=*/0,
                 llvm::MemoryBuffer::getMemBufferCopy(TestContents, TestPath));
  auto FailableFS = llvm::makeIntrusiveRefCnt<FailableProxyFS>(std::move(InMem));
  auto FileMgr = llvm::makeIntrusiveRefCnt<FileManager>(
      FileSystemOptions(), FailableFS);

  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags(new DiagnosticsEngine(
      DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()));
  SourceManager SrcMgr(*Diags, *FileMgr);
  Diags->setSourceManager(&SrcMgr);

  llvm::PrefixMapper Mapper;
  CachingDiagnosticsProcessor Processor(std::move(Mapper), *FileMgr);
  Processor.insertDiagConsumer(*Diags);

  // Register the file entry while reads still succeed.
  auto FE = FileMgr->getFileRef(TestPath);
  ASSERT_TRUE((bool)FE);
  FileID FID = SrcMgr.createFileID(*FE, SourceLocation(), SrcMgr::C_User);
  SourceLocation Loc = SrcMgr.getLocForStartOfFile(FID);

  // From this point, the captured diagnostic's buffer can no longer be read,
  // so the cache-miss conversion in HandleDiagnostic must fail.
  FailableFS->FailOpens = true;

  unsigned DiagID =
      Diags->getCustomDiagID(DiagnosticsEngine::Warning, "test diagnostic");
  Diags->Report(Loc, DiagID);

  // The capture failure must propagate out of serializeEmittedDiagnostics so
  // that the caller does not write an incomplete result to the cache.
  auto Serialized = Processor.serializeEmittedDiagnostics();
  EXPECT_THAT_EXPECTED(Serialized, llvm::Failed());
  Processor.removeDiagConsumer(*Diags);
}

} // anonymous namespace
