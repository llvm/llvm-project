//== unittests/Serialization/LoadSpecLazily.cpp ----------------------========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace {

class LoadSpecLazilyTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(
        sys::fs::createUniqueDirectory("load-spec-lazily-test", TestDir));
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

    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(new DiagnosticOptions());
    CreateInvocationOptions CIOpts;
    CIOpts.Diags = Diags;
    CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();

    std::string CacheBMIPath =
        llvm::Twine(TestDir + "/" + ModuleName + " .pcm").str();
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

    CompilerInstance Instance;
    Instance.setDiagnostics(Diags.get());
    Instance.setInvocation(Invocation);
    GenerateModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());

    return CacheBMIPath;
  }
};

class DeclsReaderListener : public ASTDeserializationListener {
public:
  void DeclRead(serialization::DeclID ID, const Decl *D) override {
    auto *ND = dyn_cast<NamedDecl>(D);
    if (!ND)
      return;

    EXPECT_FALSE(ND->getName().contains(ForbiddenName));
  }

  DeclsReaderListener(StringRef ForbiddenName) : ForbiddenName(ForbiddenName) {}

  StringRef ForbiddenName;
};

class LoadSpecLazilyConsumer : public ASTConsumer {
  DeclsReaderListener Listener;

public:
  LoadSpecLazilyConsumer(StringRef ForbiddenName) : Listener(ForbiddenName) {}

  ASTDeserializationListener *GetASTDeserializationListener() override {
    return &Listener;
  }
};

class CheckLoadSpecLazilyAction : public ASTFrontendAction {
  StringRef ForbiddenName;

public:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef /*Unused*/) override {
    return std::make_unique<LoadSpecLazilyConsumer>(ForbiddenName);
  }

  CheckLoadSpecLazilyAction(StringRef ForbiddenName)
      : ForbiddenName(ForbiddenName) {}
};

TEST_F(LoadSpecLazilyTest, BasicTest) {
  GenerateModuleInterface("M", R"cpp(
export module M;
export template <class T>
class A {};

export class ShouldNotBeLoaded {};

export class Temp {
   A<ShouldNotBeLoaded> AS;
};
  )cpp");

  const char *test_file_contents = R"cpp(
import M;
A<int> a;
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckLoadSpecLazilyAction>("ShouldNotBeLoaded"),
      test_file_contents,
      {
          "-std=c++20",
          DepArg.c_str(),
          "-I",
          TestDir.c_str(),
      },
      "test.cpp"));
}

TEST_F(LoadSpecLazilyTest, ChainedTest) {
  GenerateModuleInterface("M", R"cpp(
export module M;
export template <class T>
class A {};
  )cpp");

  GenerateModuleInterface("N", R"cpp(
export module N;
export import M;
export class ShouldNotBeLoaded {};

export class Temp {
   A<ShouldNotBeLoaded> AS;
};
  )cpp");

  const char *test_file_contents = R"cpp(
import N;
A<int> a;
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  EXPECT_TRUE(runToolOnCodeWithArgs(
      std::make_unique<CheckLoadSpecLazilyAction>("ShouldNotBeLoaded"),
      test_file_contents,
      {
          "-std=c++20",
          DepArg.c_str(),
          "-I",
          TestDir.c_str(),
      },
      "test.cpp"));
}

} // namespace
