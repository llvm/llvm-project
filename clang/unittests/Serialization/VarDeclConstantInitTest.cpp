//===- unittests/Serialization/VarDeclConstantInitTest.cpp - CI tests -----===//
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

class VarDeclConstantInitTest : public ::testing::Test {
  void SetUp() override {
    ASSERT_FALSE(sys::fs::createUniqueDirectory("modules-test", TestDir));
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

TEST_F(VarDeclConstantInitTest, CachedConstantInit) {
  addFile("Cached.cppm", R"cpp(
export module Fibonacci.Cache;

export namespace Fibonacci
{
	constexpr unsigned long Recursive(unsigned long n)
	{
		if (n == 0)
			return 0;
		if (n == 1)
			return 1;
		return Recursive(n - 2) + Recursive(n - 1);
	}

	template<unsigned long N>
	struct Number{};

	struct DefaultStrategy
	{
		constexpr unsigned long operator()(unsigned long n, auto... other) const
		{
			return (n + ... + other);
		}
	};

  constexpr unsigned long Compute(Number<10ul>, auto strategy)
	{
		return strategy(Recursive(10ul));
	}

	template<unsigned long N, typename Strategy = DefaultStrategy>
	constexpr unsigned long Cache = Compute(Number<N>{}, Strategy{});

  template constexpr unsigned long Cache<10ul>;
}
  )cpp");

  CreateInvocationOptions CIOpts;
  CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*CIOpts.VFS, new DiagnosticOptions());
  CIOpts.Diags = Diags;

  const char *Args[] = {"clang++",       "-std=c++20",
                        "--precompile",  "-working-directory",
                        TestDir.c_str(), "Cached.cppm"};
  std::shared_ptr<CompilerInvocation> Invocation =
      createInvocation(Args, CIOpts);
  ASSERT_TRUE(Invocation);
  Invocation->getFrontendOpts().DisableFree = false;

  CompilerInstance Instance;
  Instance.setDiagnostics(Diags.get());
  Instance.setInvocation(Invocation);

  std::string CacheBMIPath = llvm::Twine(TestDir + "/Cached.pcm").str();
  Instance.getFrontendOpts().OutputFile = CacheBMIPath;

  GenerateReducedModuleInterfaceAction Action;
  ASSERT_TRUE(Instance.ExecuteAction(Action));
  ASSERT_FALSE(Diags->hasErrorOccurred());

  std::string DepArg =
      llvm::Twine("-fmodule-file=Fibonacci.Cache=" + CacheBMIPath).str();
  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      R"cpp(
import Fibonacci.Cache;
        )cpp",
      /*Args=*/{"-std=c++20", DepArg.c_str()});

  using namespace clang::ast_matchers;
  ASTContext &Ctx = AST->getASTContext();
  const auto *cached = selectFirst<VarDecl>(
      "Cache",
      match(varDecl(isTemplateInstantiation(), hasName("Cache")).bind("Cache"),
            Ctx));
  EXPECT_TRUE(cached);
  EXPECT_TRUE(cached->getEvaluatedStmt());
  EXPECT_TRUE(cached->getEvaluatedStmt()->WasEvaluated);
  EXPECT_TRUE(cached->getEvaluatedValue());
  EXPECT_TRUE(cached->getEvaluatedValue()->isInt());
  EXPECT_EQ(cached->getEvaluatedValue()->getInt(), 55);
}

} // anonymous namespace
