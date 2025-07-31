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
    Instance.setDiagnostics(Diags.get());
    Instance.getFrontendOpts().OutputFile = CacheBMIPath;
    // Avoid memory leaks.
    Instance.getFrontendOpts().DisableFree = false;
    GenerateModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());

    return CacheBMIPath;
  }
};

enum class CheckingMode { Forbidden, Required };

class DeclsReaderListener : public ASTDeserializationListener {
  StringRef SpeficiedName;
  CheckingMode Mode;

  bool ReadedSpecifiedName = false;

public:
  void DeclRead(GlobalDeclID ID, const Decl *D) override {
    auto *ND = dyn_cast<NamedDecl>(D);
    if (!ND)
      return;

    ReadedSpecifiedName |= ND->getName().contains(SpeficiedName);
    if (Mode == CheckingMode::Forbidden) {
      EXPECT_FALSE(ReadedSpecifiedName);
    }
  }

  DeclsReaderListener(StringRef SpeficiedName, CheckingMode Mode)
      : SpeficiedName(SpeficiedName), Mode(Mode) {}

  ~DeclsReaderListener() {
    if (Mode == CheckingMode::Required) {
      EXPECT_TRUE(ReadedSpecifiedName);
    }
  }
};

class LoadSpecLazilyConsumer : public ASTConsumer {
  DeclsReaderListener Listener;

public:
  LoadSpecLazilyConsumer(StringRef SpecifiedName, CheckingMode Mode)
      : Listener(SpecifiedName, Mode) {}

  ASTDeserializationListener *GetASTDeserializationListener() override {
    return &Listener;
  }
};

class CheckLoadSpecLazilyAction : public ASTFrontendAction {
  StringRef SpecifiedName;
  CheckingMode Mode;

public:
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef /*Unused*/) override {
    return std::make_unique<LoadSpecLazilyConsumer>(SpecifiedName, Mode);
  }

  CheckLoadSpecLazilyAction(StringRef SpecifiedName, CheckingMode Mode)
      : SpecifiedName(SpecifiedName), Mode(Mode) {}
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
  EXPECT_TRUE(
      runToolOnCodeWithArgs(std::make_unique<CheckLoadSpecLazilyAction>(
                                "ShouldNotBeLoaded", CheckingMode::Forbidden),
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
  EXPECT_TRUE(
      runToolOnCodeWithArgs(std::make_unique<CheckLoadSpecLazilyAction>(
                                "ShouldNotBeLoaded", CheckingMode::Forbidden),
                            test_file_contents,
                            {
                                "-std=c++20",
                                DepArg.c_str(),
                                "-I",
                                TestDir.c_str(),
                            },
                            "test.cpp"));
}

/// Test that we won't crash due to we may invalidate the lazy specialization
/// lookup table during the loading process.
TEST_F(LoadSpecLazilyTest, ChainedTest2) {
  GenerateModuleInterface("M", R"cpp(
export module M;
export template <class T>
class A {};

export class B {};

export class C {
  A<B> D;
};
  )cpp");

  GenerateModuleInterface("N", R"cpp(
export module N;
export import M;
export class MayBeLoaded {};

export class Temp {
   A<MayBeLoaded> AS;
};

export class ExportedClass {};

export template<> class A<ExportedClass> {
   A<MayBeLoaded> AS;
   A<B>           AB;
};
  )cpp");

  const char *test_file_contents = R"cpp(
import N;
Temp T;
A<ExportedClass> a;
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  EXPECT_TRUE(runToolOnCodeWithArgs(std::make_unique<CheckLoadSpecLazilyAction>(
                                        "MayBeLoaded", CheckingMode::Required),
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
