//===- unittests/Serialization/NoComments.cpp - CI tests -----===//
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
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

class NoComments : public ::testing::Test {
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
};

TEST_F(NoComments, NonModulesTest) {
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      R"cpp(
/// Any comments
void foo() {}
        )cpp",
      /*Args=*/{"-std=c++20"});
  EXPECT_TRUE(AST);

  ASTContext &Ctx = AST->getASTContext();

  using namespace clang::ast_matchers;
  auto *foo = selectFirst<FunctionDecl>(
      "foo", match(functionDecl(hasName("foo")).bind("foo"), Ctx));
  EXPECT_TRUE(foo);

  const RawComment *RC = getCompletionComment(Ctx, foo);
  EXPECT_TRUE(RC);
  EXPECT_TRUE(RC->getRawText(Ctx.getSourceManager()).trim() ==
              "/// Any comments");
}

TEST_F(NoComments, ModulesTest) {
  addFile("Comments.cppm", R"cpp(
export module Comments;

/// Any comments
void foo() {}
  )cpp");

  CreateInvocationOptions CIOpts;
  CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
  CIOpts.Diags = Diags;

  std::string CacheBMIPath = llvm::Twine(TestDir + "/Comments.pcm").str();
  const char *Args[] = {"clang++",       "-std=c++20",
                        "--precompile",  "-working-directory",
                        TestDir.c_str(), "Comments.cppm"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(Args, CIOpts);
  ASSERT_TRUE(Invocation);

  CompilerInstance Instance(std::move(Invocation));
  Instance.setDiagnostics(Diags.get());
  Instance.getFrontendOpts().OutputFile = CacheBMIPath;
  GenerateReducedModuleInterfaceAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  std::string DepArg =
      llvm::Twine("-fmodule-file=Comments=" + CacheBMIPath).str();
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      R"cpp(
import Comments;
        )cpp",
      /*Args=*/{"-std=c++20", DepArg.c_str()});
  EXPECT_TRUE(AST);

  ASTContext &Ctx = AST->getASTContext();

  using namespace clang::ast_matchers;
  auto *foo = selectFirst<FunctionDecl>(
      "foo", match(functionDecl(hasName("foo")).bind("foo"), Ctx));
  EXPECT_TRUE(foo);

  const RawComment *RC = getCompletionComment(Ctx, foo);
  EXPECT_FALSE(RC);
}

} // anonymous namespace
