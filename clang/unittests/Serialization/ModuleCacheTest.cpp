//===- unittests/Serialization/ModuleCacheTest.cpp - CI tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ModuleCache.h"
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/FileManager.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#ifndef _WIN32
#include <csignal>
#include <sys/resource.h>
#endif

using namespace llvm;
using namespace clang;

namespace {

class ModuleCacheTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("modulecache-test", TestDir));

    ModuleCachePath = SmallString<256>(TestDir);
    sys::path::append(ModuleCachePath, ModuleCacheDir);
    ASSERT_FALSE(sys::fs::create_directories(ModuleCachePath));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

public:
  SmallString<256> TestDir;
  SmallString<256> ModuleCachePath;
  static constexpr StringRef ModuleCacheDir = "mcp";

  void addFile(StringRef Path, StringRef Contents) {
    ASSERT_FALSE(sys::path::is_absolute(Path));

    SmallString<256> AbsPath(TestDir);
    sys::path::append(AbsPath, Path);

    std::error_code EC;
    ASSERT_FALSE(
        sys::fs::create_directories(llvm::sys::path::parent_path(AbsPath)));
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  // Checks that no temporary files were left behind in the module cache.
  void expectNoTemporaryFiles() {
    std::error_code EC;
    for (sys::fs::recursive_directory_iterator I(ModuleCachePath, EC), E;
         I != E && !EC; I.increment(EC))
      EXPECT_NE(sys::path::extension(I->path()), ".tmp") << I->path();
    EXPECT_FALSE(EC);
  }

  void addDuplicateFrameworks() {
    addFile("test.m", R"cpp(
        @import Top;
    )cpp");

    addFile("frameworks/Top.framework/Headers/top.h", R"cpp(
        @import M;
    )cpp");
    addFile("frameworks/Top.framework/Modules/module.modulemap", R"cpp(
        framework module Top [system] {
          header "top.h"
          export *
        }
    )cpp");

    addFile("frameworks/M.framework/Headers/m.h", R"cpp(
        void foo();
    )cpp");
    addFile("frameworks/M.framework/Modules/module.modulemap", R"cpp(
        framework module M [system] {
          header "m.h"
          export *
        }
    )cpp");

    addFile("frameworks2/M.framework/Headers/m.h", R"cpp(
        void foo();
    )cpp");
    addFile("frameworks2/M.framework/Modules/module.modulemap", R"cpp(
        framework module M [system] {
          header "m.h"
          export *
        }
    )cpp");
  }

  std::unique_ptr<CompilerInvocation>
  createInvocationAndEnableFree(ArrayRef<const char *> Args,
                                CreateInvocationOptions Opts) {
    std::unique_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, Opts);
    if (Invocation)
      Invocation->getFrontendOpts().DisableFree = false;

    return Invocation;
  }
};

TEST_F(ModuleCacheTest, CachedModuleNewPath) {
  addDuplicateFrameworks();

  SmallString<256> MCPArg("-fmodules-cache-path=");
  MCPArg.append(ModuleCachePath);
  CreateInvocationOptions CIOpts;
  CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
  CIOpts.Diags = Diags;

  // First run should pass with no errors
  const char *Args[] = {"clang",        "-fmodules",          "-Fframeworks",
                        MCPArg.c_str(), "-working-directory", TestDir.c_str(),
                        "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocationAndEnableFree(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  CompilerInstance Instance(std::move(Invocation));
  Instance.setVirtualFileSystem(CIOpts.VFS);
  Instance.setDiagnostics(Diags);
  SyntaxOnlyAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  // Now add `frameworks2` to the search path. `Top.pcm` will have a reference
  // to the `M` from `frameworks`, but a search will find the `M` from
  // `frameworks2` - causing a mismatch and it to be considered out of date.
  //
  // Normally this would be fine - `M` and the modules it depends on would be
  // rebuilt. However, since we have a shared module cache and thus an already
  // finalized `Top`, recompiling `Top` will cause the existing module to be
  // removed from the cache, causing possible crashed if it is ever used.
  //
  // Make sure that an error occurs instead.
  const char *Args2[] = {"clang",         "-fmodules",    "-Fframeworks2",
                         "-Fframeworks",  MCPArg.c_str(), "-working-directory",
                         TestDir.c_str(), "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation2 =
      createInvocationAndEnableFree(Args2, CIOpts);
  ASSERT_TRUE(Invocation2);
  CompilerInstance Instance2(std::move(Invocation2),
                             Instance.getPCHContainerOperations(),
                             Instance.getModuleCachePtr());
  Instance2.setVirtualFileSystem(CIOpts.VFS);
  Instance2.setDiagnostics(Diags);
  SyntaxOnlyAction Action2;
  ASSERT_FALSE(Instance2.ExecuteAction(Action2));
  ASSERT_TRUE(Diags->hasErrorOccurred());
}

TEST_F(ModuleCacheTest, CachedModuleNewPathAllowErrors) {
  addDuplicateFrameworks();

  SmallString<256> MCPArg("-fmodules-cache-path=");
  MCPArg.append(ModuleCachePath);
  CreateInvocationOptions CIOpts;
  CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
  CIOpts.Diags = Diags;

  // First run should pass with no errors
  const char *Args[] = {"clang",        "-fmodules",          "-Fframeworks",
                        MCPArg.c_str(), "-working-directory", TestDir.c_str(),
                        "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocationAndEnableFree(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  CompilerInstance Instance(std::move(Invocation));
  Instance.setVirtualFileSystem(CIOpts.VFS);
  Instance.setDiagnostics(Diags);
  SyntaxOnlyAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  // Same as `CachedModuleNewPath` but while allowing errors. This is a hard
  // failure where the module wasn't created, so it should still fail.
  const char *Args2[] = {
      "clang",         "-fmodules",    "-Fframeworks2",
      "-Fframeworks",  MCPArg.c_str(), "-working-directory",
      TestDir.c_str(), "-Xclang",      "-fallow-pcm-with-compiler-errors",
      "test.m"};
  std::shared_ptr<CompilerInvocation> Invocation2 =
      createInvocationAndEnableFree(Args2, CIOpts);
  ASSERT_TRUE(Invocation2);
  CompilerInstance Instance2(std::move(Invocation2),
                             Instance.getPCHContainerOperations(),
                             Instance.getModuleCachePtr());
  Instance2.setVirtualFileSystem(CIOpts.VFS);
  Instance2.setDiagnostics(Diags);
  SyntaxOnlyAction Action2;
  ASSERT_FALSE(Instance2.ExecuteAction(Action2));
  ASSERT_TRUE(Diags->hasErrorOccurred());
}

struct CapturedDiag {
  unsigned ID;
  std::string Message;
};

class CapturedDiagConsumer : public DiagnosticConsumer {
  std::vector<CapturedDiag> &Diags;

public:
  CapturedDiagConsumer(std::vector<CapturedDiag> &D) : Diags(D) {}
  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic &Info) override {
    DiagnosticConsumer::HandleDiagnostic(Level, Info);
    SmallString<128> Msg;
    Info.FormatDiagnostic(Msg);
    Diags.push_back({Info.getID(), std::string(Msg)});
  }
};

TEST_F(ModuleCacheTest, RebuildFinalizedModuleAfterInputChange) {
  addFile("test.m", R"cpp(
          @import M;
      )cpp");
  addFile("frameworks/M.framework/Headers/m.h", R"cpp(
          void foo(void);
      )cpp");
  addFile("frameworks/M.framework/Modules/module.modulemap", R"cpp(
          framework module M [system] {
            header "m.h"
            export *
          }
      )cpp");

  SmallString<256> MCPArg("-fmodules-cache-path=");
  MCPArg.append(ModuleCachePath);
  CreateInvocationOptions CIOpts;
  CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
  CIOpts.Diags = Diags;

  const char *Args[] = {"clang",        "-fmodules",          "-Fframeworks",
                        MCPArg.c_str(), "-working-directory", TestDir.c_str(),
                        "test.m"};

  // Run 1: builds, loads, and finalizes M in the shared in-memory cache.
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocationAndEnableFree(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  CompilerInstance Instance(std::move(Invocation));
  Instance.setVirtualFileSystem(CIOpts.VFS);
  Instance.setDiagnostics(Diags);
  SyntaxOnlyAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  // Grow m.h so its recorded size no longer matches, marking M out of date on
  // the next load. Because M is finalized in the shared cache, it can be
  // neither dropped nor rebuilt.
  addFile("frameworks/M.framework/Headers/m.h", R"cpp(
          void foo(void);
          void bar(void);
          void baz(void);
      )cpp");

  std::vector<CapturedDiag> Captured;
  Diags->setClient(new CapturedDiagConsumer(Captured),
                   /*ShouldOwnClient=*/true);

  // Run 2 shares the module cache (and thus the finalized M).
  std::shared_ptr<CompilerInvocation> Invocation2 =
      createInvocationAndEnableFree(Args, CIOpts);
  ASSERT_TRUE(Invocation2);
  CompilerInstance Instance2(std::move(Invocation2),
                             Instance.getPCHContainerOperations(),
                             Instance.getModuleCachePtr());
  Instance2.setVirtualFileSystem(CIOpts.VFS);
  Instance2.setDiagnostics(Diags);
  SyntaxOnlyAction Action2;
  ASSERT_FALSE(Instance2.ExecuteAction(Action2));
  ASSERT_TRUE(Diags->hasErrorOccurred());

  ASSERT_EQ(Captured.size(), 4u);
  EXPECT_EQ(Captured[0].ID, (unsigned)diag::err_fe_ast_file_modified);
  EXPECT_NE(Captured[0].Message.find("m.h"), std::string::npos);
  EXPECT_EQ(Captured[1].ID, (unsigned)diag::note_fe_ast_file_modified);
  EXPECT_EQ(Captured[2].ID,
            (unsigned)diag::note_fe_ast_file_modified_finalized);
  EXPECT_NE(Captured[2].Message.find("'M'"), std::string::npos);
  EXPECT_EQ(Captured[3].ID,
            (unsigned)diag::note_ast_file_input_files_validation_status);
}

// Test that even for a small file not mmapped, the access time is still updated
// upon read. Accurate atime is necessary for clang's pruning logic.
#ifndef _WIN32
TEST_F(ModuleCacheTest, ReadRefreshesAccessTime) {
  SmallString<256> RelPath(ModuleCacheDir);
  sys::path::append(RelPath, "Small.pcm");
  addFile(RelPath, "not really a module file");
  SmallString<256> Path(TestDir);
  sys::path::append(Path, RelPath);

  using namespace std::chrono;
  sys::TimePoint<> Now = time_point_cast<seconds>(system_clock::now());
  sys::TimePoint<> OldModTime = Now - hours(24 * 10);
  sys::TimePoint<> OldAccessTime = Now - hours(2);
  ASSERT_FALSE(sys::fs::setLastAccessAndModificationTime(Path, OldAccessTime,
                                                         OldModTime));

  off_t Size;
  time_t ModTime;
  Expected<std::unique_ptr<MemoryBuffer>> Buf = readImpl(Path, Size, ModTime);
  ASSERT_TRUE(static_cast<bool>(Buf));
  EXPECT_EQ(Size, 24);
  EXPECT_EQ(ModTime, sys::toTimeT(OldModTime));

  sys::fs::file_status Status;
  ASSERT_FALSE(sys::fs::status(Path, Status));
  EXPECT_GE(Status.getLastAccessedTime(), Now);
  EXPECT_EQ(Status.getLastModificationTime(), OldModTime);
}
#endif

// Test that the reported size covers the entire written contents, even if the
// stream hasn't flushed them to the file on its own.
TEST_F(ModuleCacheTest, WriteReportsFullSize) {
  SmallString<256> Path(ModuleCachePath);
  sys::path::append(Path, "M.pcm");

  StringRef Contents = "contents\n";
  off_t Size;
  time_t ModTime;
  ASSERT_EQ(writeImpl(Path, MemoryBufferRef(Contents, Path), Size, ModTime,
                      /*ReadBack=*/false)
                .getError(),
            std::error_code{});
  EXPECT_EQ(Size, off_t(Contents.size()));
}

// Test that the buffer returned from write() is memory-mapped to the written
// file (for files large enough to be mmapped).
TEST_F(ModuleCacheTest, WriteReturnsMappedBuffer) {
  SmallString<256> Path(ModuleCachePath);
  sys::path::append(Path, "M.pcm");

  std::string Contents(1024 * 1024, 'x');
  auto Given = MemoryBuffer::getMemBuffer(Contents, Path);
  ASSERT_NE(Given->getBufferKind(), MemoryBuffer::MemoryBuffer_MMap);

  off_t Size;
  time_t ModTime;
  std::shared_ptr<ModuleCache> ModCache = createCrossProcessModuleCache();
  auto WrittenOrErr = ModCache->write(Path, std::move(Given), Size, ModTime);
  ASSERT_EQ(WrittenOrErr.getError(), std::error_code{});
  auto Written = std::move(*WrittenOrErr);
  ASSERT_TRUE(Written);
  // Written buffer refers to the mmaped file contents.
  EXPECT_EQ(Written->getBufferKind(), MemoryBuffer::MemoryBuffer_MMap);
  EXPECT_EQ(Written->getBuffer(), Contents);
}

#ifndef _WIN32
static std::error_code writeError(StringRef Path, StringRef Contents,
                                  bool ReadBack = true) {
  off_t Size;
  time_t ModTime;
  return writeImpl(Path, MemoryBufferRef(Contents, Path), Size, ModTime,
                   ReadBack)
      .getError();
}

// Test that failing to create the temporary file for a reason other than a
// missing parent directory is reported.
TEST_F(ModuleCacheTest, WriteErrorCreateTempFile) {
  // The parent "directory" is a regular file.
  SmallString<256> RelPath(ModuleCacheDir);
  sys::path::append(RelPath, "file");
  addFile(RelPath, "");
  SmallString<256> Path(TestDir);
  sys::path::append(Path, RelPath, "M.pcm");

  EXPECT_EQ(writeError(Path, "contents"), std::errc::not_a_directory);
}

// Test that failing to create the missing parent directory is reported.
TEST_F(ModuleCacheTest, WriteErrorCreateDirectories) {
  // Creating the temporary file in "dangling/sub" fails because "sub" doesn't
  // exist. Creating "dangling/sub" then fails, because "dangling" is a dangling
  // symlink and it's not possible to create a directory inside of it.
  SmallString<256> Dangling(ModuleCachePath);
  sys::path::append(Dangling, "dangling");
  ASSERT_FALSE(sys::fs::create_link("does-not-exist", Dangling));
  SmallString<256> Path(Dangling);
  sys::path::append(Path, "sub", "M.pcm");

  EXPECT_EQ(writeError(Path, "contents"), std::errc::no_such_file_or_directory);
}

// Test that failing to create the temporary file after creating the missing
// parent directory is reported.
TEST_F(ModuleCacheTest, WriteErrorCreateTempFileAfterCreateDirectories) {
  // Creating the temporary file in "dangling" fails because it's a dangling
  // symlink. Creating the directory "dangling" then succeeds, because it
  // already exists (as a symlink), but creating the temporary file again fails.
  SmallString<256> Dangling(ModuleCachePath);
  sys::path::append(Dangling, "dangling");
  ASSERT_FALSE(sys::fs::create_link("does-not-exist", Dangling));
  SmallString<256> Path(Dangling);
  sys::path::append(Path, "M.pcm");

  EXPECT_EQ(writeError(Path, "contents"), std::errc::no_such_file_or_directory);
}

// Test that failing to write the contents is reported instead of crashing in
// the destructor of the stream, and that the temporary file is removed.
TEST_F(ModuleCacheTest, WriteErrorWrite) {
  SmallString<256> Path(ModuleCachePath);
  sys::path::append(Path, "M.pcm");

  // Limit the file size, making writes past the limit fail with EFBIG instead
  // of delivering SIGXFSZ.
  struct rlimit OldLimit;
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &OldLimit), 0);
  struct rlimit NewLimit = OldLimit;
  NewLimit.rlim_cur = 1024;
  auto *OldHandler = std::signal(SIGXFSZ, SIG_IGN);
  ASSERT_NE(OldHandler, SIG_ERR);
  ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &NewLimit), 0);
  llvm::scope_exit Restore([&] {
    setrlimit(RLIMIT_FSIZE, &OldLimit);
    std::signal(SIGXFSZ, OldHandler);
  });

  std::string Contents(1024 * 1024, 'x');
  EXPECT_EQ(writeError(Path, Contents, /*ReadBack=*/true),
            std::errc::file_too_large);
  EXPECT_EQ(writeError(Path, Contents, /*ReadBack=*/false),
            std::errc::file_too_large);
  EXPECT_FALSE(sys::fs::exists(Path));
  expectNoTemporaryFiles();
}

// Test that failing to rename the temporary file to the final path is
// reported, and that the temporary file is removed.
TEST_F(ModuleCacheTest, WriteErrorRename) {
  // The final path is a non-empty directory.
  SmallString<256> RelPath(ModuleCacheDir);
  sys::path::append(RelPath, "M.pcm", "file");
  addFile(RelPath, "");
  SmallString<256> Path(ModuleCachePath);
  sys::path::append(Path, "M.pcm");

  EXPECT_EQ(writeError(Path, "contents"), std::errc::is_a_directory);
  expectNoTemporaryFiles();
}
#endif

} // anonymous namespace
