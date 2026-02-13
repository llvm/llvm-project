//== unittests/Serialization/NamespaceLookupOptimizationTest.cpp =======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace {

class NamespaceLookupTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(
        sys::fs::createUniqueDirectory("namespace-lookup-test", TestDir));
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

public:
  SmallString<256> TestDir;

  void addFile(StringRef Path, StringRef Contents) {
    ASSERT_FALSE(sys::path::is_absolute(Path));

    SmallString<256> AbsPath(TestDir);
    sys::path::append(AbsPath, Path);

    ASSERT_FALSE(
        sys::fs::create_directories(llvm::sys::path::parent_path(AbsPath)));

    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  std::string GenerateModuleInterface(StringRef ModuleName,
                                      StringRef Contents) {
    std::string FileName = llvm::Twine(ModuleName + ".cppm").str();
    addFile(FileName, Contents);

    IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS =
        llvm::vfs::createPhysicalFileSystem();
    DiagnosticOptions DiagOpts;
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*VFS, DiagOpts);
    CreateInvocationOptions CIOpts;
    CIOpts.Diags = Diags;
    CIOpts.VFS = VFS;

    std::string CacheBMIPath =
        llvm::Twine(TestDir + "/" + ModuleName + ".pcm").str();
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
                          FileName.c_str(),
                          "-o",
                          CacheBMIPath.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    EXPECT_TRUE(Invocation);

    CompilerInstance Instance(std::move(Invocation));
    Instance.setDiagnostics(Diags);
    Instance.getFrontendOpts().OutputFile = CacheBMIPath;
    // Avoid memory leaks.
    Instance.getFrontendOpts().DisableFree = false;
    GenerateModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());

    return CacheBMIPath;
  }
};

struct NamespaceLookupResult {
  int NumLocalNamespaces = 0;
  int NumExternalNamespaces = 0;
};

class NamespaceLookupConsumer : public ASTConsumer {
  NamespaceLookupResult &Result;

public:
  explicit NamespaceLookupConsumer(NamespaceLookupResult &Result)
      : Result(Result) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    TranslationUnitDecl *TU = Context.getTranslationUnitDecl();
    ASSERT_TRUE(TU);
    ASTReader *Chain = dyn_cast_or_null<ASTReader>(Context.getExternalSource());
    ASSERT_TRUE(Chain);
    for (const Decl *D :
         TU->lookup(DeclarationName(&Context.Idents.get("N")))) {
      if (!isa<NamespaceDecl>(D))
        continue;
      if (!D->isFromASTFile()) {
        ++Result.NumLocalNamespaces;
      } else {
        ++Result.NumExternalNamespaces;
        EXPECT_EQ(D, Chain->getKeyDeclaration(D));
      }
    }
  }
};

class NamespaceLookupAction : public ASTFrontendAction {
  NamespaceLookupResult &Result;

public:
  explicit NamespaceLookupAction(NamespaceLookupResult &Result)
      : Result(Result) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef /*Unused*/) override {
    return std::make_unique<NamespaceLookupConsumer>(Result);
  }
};

TEST_F(NamespaceLookupTest, ExternalNamespacesOnly) {
  GenerateModuleInterface("M1", R"cpp(
export module M1;
namespace N {}
  )cpp");
  GenerateModuleInterface("M2", R"cpp(
export module M2;
namespace N {}
  )cpp");
  GenerateModuleInterface("M3", R"cpp(
export module M3;
namespace N {}
  )cpp");
  const char *test_file_contents = R"cpp(
import M1;
import M2;
import M3;
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  NamespaceLookupResult Result;
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<NamespaceLookupAction>(Result), test_file_contents,
      {
          "-std=c++20",
          DepArg.c_str(),
          "-I",
          TestDir.c_str(),
      },
      "main.cpp"));

  EXPECT_EQ(0, Result.NumLocalNamespaces);
  EXPECT_EQ(1, Result.NumExternalNamespaces);
}

TEST_F(NamespaceLookupTest, ExternalReplacedByLocal) {
  GenerateModuleInterface("M1", R"cpp(
export module M1;
namespace N {}
  )cpp");
  GenerateModuleInterface("M2", R"cpp(
export module M2;
namespace N {}
  )cpp");
  GenerateModuleInterface("M3", R"cpp(
export module M3;
namespace N {}
  )cpp");
  const char *test_file_contents = R"cpp(
import M1;
import M2;
import M3;

namespace N {}
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  NamespaceLookupResult Result;
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<NamespaceLookupAction>(Result), test_file_contents,
      {
          "-std=c++20",
          DepArg.c_str(),
          "-I",
          TestDir.c_str(),
      },
      "main.cpp"));

  EXPECT_EQ(1, Result.NumLocalNamespaces);
  EXPECT_EQ(0, Result.NumExternalNamespaces);
}

TEST_F(NamespaceLookupTest, LocalAndExternalInterleaved) {
  GenerateModuleInterface("M1", R"cpp(
export module M1;
namespace N {}
  )cpp");
  GenerateModuleInterface("M2", R"cpp(
export module M2;
namespace N {}
  )cpp");
  GenerateModuleInterface("M3", R"cpp(
export module M3;
namespace N {}
  )cpp");
  const char *test_file_contents = R"cpp(
import M1;

namespace N {}

import M2;
import M3;
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  NamespaceLookupResult Result;
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<NamespaceLookupAction>(Result), test_file_contents,
      {
          "-std=c++20",
          DepArg.c_str(),
          "-I",
          TestDir.c_str(),
      },
      "main.cpp"));

  EXPECT_EQ(1, Result.NumLocalNamespaces);
  EXPECT_EQ(1, Result.NumExternalNamespaces);
}

} // namespace
