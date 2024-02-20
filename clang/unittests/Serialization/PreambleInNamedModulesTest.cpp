//===- unittests/Serialization/PreambleInNamedModulesTest.cpp -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class PreambleInNamedModulesTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("modules-test", TestDir));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

public:
  using PathType = SmallString<256>;

  PathType TestDir;

  void addFile(StringRef Path, StringRef Contents, PathType &AbsPath) {
    ASSERT_FALSE(sys::path::is_absolute(Path));

    AbsPath = TestDir;
    sys::path::append(AbsPath, Path);

    ASSERT_FALSE(
        sys::fs::create_directories(llvm::sys::path::parent_path(AbsPath)));

    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  void addFile(StringRef Path, StringRef Contents) {
    PathType UnusedAbsPath;
    addFile(Path, Contents, UnusedAbsPath);
  }
};

// Testing that the use of Preamble in named modules can work basically.
// See https://github.com/llvm/llvm-project/issues/80570
TEST_F(PreambleInNamedModulesTest, BasicTest) {
  addFile("foo.h", R"cpp(
enum class E {
    A,
    B,
    C,
    D
};
  )cpp");

  PathType MainFilePath;
  addFile("A.cppm", R"cpp(
module;
#include "foo.h"
export module A;
export using ::E;
  )cpp",
          MainFilePath);

  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS =
      llvm::vfs::createPhysicalFileSystem();

  CreateInvocationOptions CIOpts;
  CIOpts.Diags = Diags;
  CIOpts.VFS = VFS;

  const char *Args[] = {"clang++", "-std=c++20", "-working-directory",
                        TestDir.c_str(), MainFilePath.c_str()};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(Args, CIOpts);
  ASSERT_TRUE(Invocation);

  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> ContentsBuffer =
      llvm::MemoryBuffer::getFile(MainFilePath, /*IsText=*/true);
  EXPECT_TRUE(ContentsBuffer);
  std::unique_ptr<MemoryBuffer> Buffer = std::move(*ContentsBuffer);

  PreambleBounds Bounds =
      ComputePreambleBounds(Invocation->getLangOpts(), *Buffer, 0);

  PreambleCallbacks Callbacks;
  llvm::ErrorOr<PrecompiledPreamble> BuiltPreamble = PrecompiledPreamble::Build(
      *Invocation, Buffer.get(), Bounds, *Diags, VFS,
      std::make_shared<PCHContainerOperations>(),
      /*StoreInMemory=*/false, /*StoragePath=*/TestDir, Callbacks);

  ASSERT_FALSE(Diags->hasErrorOccurred());

  EXPECT_TRUE(BuiltPreamble);
  EXPECT_TRUE(BuiltPreamble->CanReuse(*Invocation, *Buffer, Bounds, *VFS));
  BuiltPreamble->OverridePreamble(*Invocation, VFS, Buffer.get());

  auto Clang = std::make_unique<CompilerInstance>(
      std::make_shared<PCHContainerOperations>());
  Clang->setInvocation(std::move(Invocation));
  Clang->setDiagnostics(Diags.get());

  if (auto VFSWithRemapping = createVFSFromCompilerInvocation(
          Clang->getInvocation(), Clang->getDiagnostics(), VFS))
    VFS = VFSWithRemapping;

  Clang->createFileManager(VFS);
  EXPECT_TRUE(Clang->createTarget());

  Buffer.release();

  SyntaxOnlyAction Action;
  EXPECT_TRUE(Clang->ExecuteAction(Action));
  EXPECT_FALSE(Clang->getDiagnosticsPtr()->hasErrorOccurred());
}

} // namespace
