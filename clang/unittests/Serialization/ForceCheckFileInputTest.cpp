//===- unittests/Serialization/ForceCheckFileInputTest.cpp - CI tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class ForceCheckFileInputTest : public ::testing::Test {
  void SetUp() override {
    EXPECT_FALSE(sys::fs::createUniqueDirectory("modules-test", TestDir));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

public:
  SmallString<256> TestDir;

  void addFile(StringRef Path, StringRef Contents) {
    EXPECT_FALSE(sys::path::is_absolute(Path));

    SmallString<256> AbsPath(TestDir);
    sys::path::append(AbsPath, Path);

    EXPECT_FALSE(
        sys::fs::create_directories(llvm::sys::path::parent_path(AbsPath)));

    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    EXPECT_FALSE(EC);
    OS << Contents;
  }
};

TEST_F(ForceCheckFileInputTest, ForceCheck) {
  addFile("a.cppm", R"cpp(
export module a;
export int aa = 43;
  )cpp");

  std::string BMIPath = llvm::Twine(TestDir + "/a.pcm").str();

  {
    CreateInvocationOptions CIOpts;
    CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();

    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*CIOpts.VFS,
                                            new DiagnosticOptions());
    CIOpts.Diags = Diags;

    const char *Args[] = {"clang++",       "-std=c++20",
                          "--precompile",  "-working-directory",
                          TestDir.c_str(), "a.cppm"};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    EXPECT_TRUE(Invocation);
    Invocation->getFrontendOpts().DisableFree = false;

    auto Buf = CIOpts.VFS->getBufferForFile("a.cppm");
    EXPECT_TRUE(Buf);

    Invocation->getPreprocessorOpts().addRemappedFile("a.cppm", Buf->get());

    Buf->release();

    CompilerInstance Instance(std::move(Invocation));
    Instance.setDiagnostics(Diags.get());

    Instance.getFrontendOpts().OutputFile = BMIPath;

    if (auto VFSWithRemapping = createVFSFromCompilerInvocation(
            Instance.getInvocation(), Instance.getDiagnostics(), CIOpts.VFS))
      CIOpts.VFS = VFSWithRemapping;
    Instance.createFileManager(CIOpts.VFS);

    Instance.getHeaderSearchOpts().ValidateASTInputFilesContent = true;

    GenerateReducedModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());
  }

  {
    CreateInvocationOptions CIOpts;
    CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*CIOpts.VFS,
                                            new DiagnosticOptions());
    CIOpts.Diags = Diags;

    std::string BMIPath = llvm::Twine(TestDir + "/a.pcm").str();
    const char *Args[] = {
        "clang++",       "-std=c++20", "--precompile", "-working-directory",
        TestDir.c_str(), "a.cppm",     "-o",           BMIPath.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    EXPECT_TRUE(Invocation);
    Invocation->getFrontendOpts().DisableFree = false;

    CompilerInstance Clang(std::move(Invocation));

    Clang.setDiagnostics(Diags.get());
    FileManager *FM = Clang.createFileManager(CIOpts.VFS);
    Clang.createSourceManager(*FM);

    EXPECT_TRUE(Clang.createTarget());
    Clang.createPreprocessor(TU_Complete);
    Clang.getHeaderSearchOpts().ForceCheckCXX20ModulesInputFiles = true;
    Clang.getHeaderSearchOpts().ValidateASTInputFilesContent = true;
    Clang.createASTReader();

    addFile("a.cppm", R"cpp(
export module a;
export int aa = 44;
  )cpp");

    auto ReadResult =
        Clang.getASTReader()->ReadAST(BMIPath, serialization::MK_MainFile,
                                      SourceLocation(), ASTReader::ARR_None);

    // We shall be able to detect the content change here.
    EXPECT_NE(ReadResult, ASTReader::Success);
  }
}

} // anonymous namespace
