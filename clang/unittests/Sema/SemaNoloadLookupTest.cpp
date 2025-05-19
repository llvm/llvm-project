//== unittests/Sema/SemaNoloadLookupTest.cpp -------------------------========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclarationName.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaConsumer.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace {

class NoloadLookupTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(
        sys::fs::createUniqueDirectory("modules-no-comments-test", TestDir));
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

    CreateInvocationOptions CIOpts;
    CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*CIOpts.VFS,
                                            new DiagnosticOptions());
    CIOpts.Diags = Diags;

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
                          FileName.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    EXPECT_TRUE(Invocation);

    CompilerInstance Instance(std::move(Invocation));
    Instance.setDiagnostics(Diags.get());
    Instance.getFrontendOpts().OutputFile = CacheBMIPath;
    GenerateReducedModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());

    return CacheBMIPath;
  }
};

struct TrivialVisibleDeclConsumer : public VisibleDeclConsumer {
  TrivialVisibleDeclConsumer() {}
  void EnteredContext(DeclContext *Ctx) override {}
  void FoundDecl(NamedDecl *ND, NamedDecl *Hiding, DeclContext *Ctx,
                 bool InBaseClass) override {
    FoundNum++;
  }

  int FoundNum = 0;
};

class NoloadLookupConsumer : public SemaConsumer {
public:
  void InitializeSema(Sema &S) override { SemaPtr = &S; }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    if (!D.isSingleDecl())
      return true;

    Decl *TD = D.getSingleDecl();

    auto *ID = dyn_cast<ImportDecl>(TD);
    if (!ID)
      return true;

    clang::Module *M = ID->getImportedModule();
    assert(M);
    if (M->Name != "R")
      return true;

    auto *Std = SemaPtr->getStdNamespace();
    EXPECT_TRUE(Std);
    TrivialVisibleDeclConsumer Consumer;
    SemaPtr->LookupVisibleDecls(Std, Sema::LookupNameKind::LookupOrdinaryName,
                                Consumer,
                                /*IncludeGlobalScope=*/true,
                                /*IncludeDependentBases=*/false,
                                /*LoadExternal=*/false);
    EXPECT_EQ(Consumer.FoundNum, 1);
    return true;
  }

private:
  Sema *SemaPtr = nullptr;
};

class NoloadLookupAction : public ASTFrontendAction {
  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef /*Unused*/) override {
    return std::make_unique<NoloadLookupConsumer>();
  }
};

TEST_F(NoloadLookupTest, NonModulesTest) {
  GenerateModuleInterface("M", R"cpp(
module;
namespace std {
  int What();

  void bar(int x = What()) {
  }
}
export module M;
export using std::bar;
  )cpp");

  GenerateModuleInterface("R", R"cpp(
module;
namespace std {
  class Another;
  int What(Another);
  int What();
}
export module R;
  )cpp");

  const char *test_file_contents = R"cpp(
import M;
namespace std {
  void use() {
    bar();
  }
}
import R;
  )cpp";
  std::string DepArg = "-fprebuilt-module-path=" + TestDir.str().str();
  EXPECT_TRUE(runToolOnCodeWithArgs(std::make_unique<NoloadLookupAction>(),
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
