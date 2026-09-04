//===- unittests/Sema/SemaModuleTest.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Decl.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

namespace {

class SemaModuleTest : public ::testing::Test {
protected:
  SmallString<256> TestDir;

  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("sema-module-test", TestDir));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

  void addFile(StringRef Path, StringRef Contents) {
    ASSERT_FALSE(sys::path::is_absolute(Path));

    SmallString<256> AbsPath(TestDir);
    sys::path::append(AbsPath, Path);
    ASSERT_FALSE(sys::fs::create_directories(sys::path::parent_path(AbsPath)));

    std::error_code EC;
    raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  void generateModuleInterface(StringRef ModuleName, StringRef Contents) {
    std::string FileName = (ModuleName + ".cppm").str();
    addFile(FileName, Contents);

    CreateInvocationOptions CIOpts;
    CIOpts.VFS = vfs::createPhysicalFileSystem();
    DiagnosticOptions DiagOpts;
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
    CIOpts.Diags = Diags;

    std::string BMIPath = (TestDir + "/" + ModuleName + ".pcm").str();
    std::string PrebuiltModulePath =
        "-fprebuilt-module-path=" + TestDir.str().str();
    const char *Args[] = {"clang++",
                          "-std=c++20",
                          "--precompile",
                          PrebuiltModulePath.c_str(),
                          "-working-directory",
                          TestDir.c_str(),
                          "-I",
                          TestDir.c_str(),
                          FileName.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    ASSERT_TRUE(Invocation);

    CompilerInstance Instance(std::move(Invocation));
    Instance.setDiagnostics(Diags);
    Instance.getFrontendOpts().OutputFile = BMIPath;
    Instance.getFrontendOpts().DisableFree = false;
    GenerateModuleInterfaceAction Action;
    ASSERT_TRUE(Instance.ExecuteAction(Action));
    ASSERT_FALSE(Diags->hasErrorOccurred());
  }
};

struct MergedEnumResult {
  bool FoundEnumerator = false;
};

class MergedEnumConsumer : public ASTConsumer {
public:
  explicit MergedEnumConsumer(MergedEnumResult &Result) : Result(Result) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    for (Decl *D : Context.getTranslationUnitDecl()->decls()) {
      auto *ED = dyn_cast<EnumDecl>(D);
      if (!ED || ED->isFromASTFile())
        continue;

      for (EnumConstantDecl *ECD : ED->enumerators()) {
        if (ECD->getName() != "TypedefValue")
          continue;

        ASSERT_FALSE(Result.FoundEnumerator);
        Result.FoundEnumerator = true;

        ASSERT_EQ(ECD->getDeclContext(), ED);
        EXPECT_EQ(ECD->getType(), Context.getCanonicalTagType(ED));
        ASSERT_TRUE(ED->getPreviousDecl());
        EXPECT_TRUE(ED->getPreviousDecl()->isFromASTFile());
        EXPECT_EQ(ECD->getType(),
                  Context.getCanonicalTagType(ED->getPreviousDecl()));
      }
    }
  }

private:
  MergedEnumResult &Result;
};

class MergedEnumAction : public ASTFrontendAction {
public:
  explicit MergedEnumAction(MergedEnumResult &Result) : Result(Result) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override {
    return std::make_unique<MergedEnumConsumer>(Result);
  }

private:
  MergedEnumResult &Result;
};

TEST_F(SemaModuleTest, MergedAnonymousEnumHasConsistentEnumeratorType) {
  addFile("enum.h", R"cpp(
#ifndef ENUM_H
#define ENUM_H
typedef enum { TypedefValue } TypedefEnum;
#endif
)cpp");
  generateModuleInterface("M", R"cpp(
module;
#include "enum.h"
export module M;
export TypedefEnum typedefEnum;
)cpp");

  std::string PrebuiltModulePath =
      "-fprebuilt-module-path=" + TestDir.str().str();
  MergedEnumResult Result;
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<MergedEnumAction>(Result), R"cpp(
import M;
#include "enum.h"
)cpp",
      {"-std=c++20", PrebuiltModulePath, "-I", TestDir.str().str()},
      "use.cpp"));
  EXPECT_TRUE(Result.FoundEnumerator);
}

} // namespace
