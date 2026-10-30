//===- unittest/Tooling/RecursiveASTVisitorTests/LambdaExprModules.cpp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

// Counts lambdas visited in the implicit-code model (the model ParentMapContext
// uses) and whether their bodies were visited too.
class ImplicitLambdaBodyVisitor
    : public RecursiveASTVisitor<ImplicitLambdaBodyVisitor> {
public:
  bool shouldVisitImplicitCode() const { return true; }

  bool VisitLambdaExpr(LambdaExpr *LE) {
    ++NumLambdas;
    Bodies.insert(LE->getBody());
    return true;
  }

  bool VisitStmt(Stmt *S) {
    if (Bodies.contains(S))
      ++NumVisitedBodies;
    return true;
  }

  unsigned NumLambdas = 0;
  unsigned NumVisitedBodies = 0;

private:
  llvm::SmallPtrSet<Stmt *, 8> Bodies;
};

class LambdaExprModulesTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("lambda-expr-modules", TestDir));
  }

  void TearDown() override { llvm::sys::fs::remove_directories(TestDir); }

  void addFile(llvm::StringRef Path, llvm::StringRef Contents) {
    ASSERT_FALSE(llvm::sys::path::is_absolute(Path));

    llvm::SmallString<256> AbsPath(TestDir);
    llvm::sys::path::append(AbsPath, Path);

    ASSERT_FALSE(llvm::sys::fs::create_directories(
        llvm::sys::path::parent_path(AbsPath)));

    std::error_code EC;
    llvm::raw_fd_ostream OS(AbsPath, EC);
    ASSERT_FALSE(EC);
    OS << Contents;
  }

  std::string precompile(llvm::StringRef FileName) {
    CreateInvocationOptions CIOpts;
    CIOpts.VFS = llvm::vfs::createPhysicalFileSystem();
    DiagnosticOptions DiagOpts;
    IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
        CompilerInstance::createDiagnostics(*CIOpts.VFS, DiagOpts);
    CIOpts.Diags = Diags;

    std::string File = FileName.str();
    const char *Args[] = {"clang++",       "-std=c++20",
                          "--precompile",  "-working-directory",
                          TestDir.c_str(), File.c_str()};
    std::shared_ptr<CompilerInvocation> Invocation =
        createInvocation(Args, CIOpts);
    EXPECT_TRUE(Invocation);
    Invocation->getFrontendOpts().DisableFree = false;

    CompilerInstance Instance(std::move(Invocation));
    Instance.setDiagnostics(Diags);

    std::string PCM = (llvm::Twine(TestDir) + "/" + File + ".pcm").str();
    Instance.getFrontendOpts().OutputFile = PCM;

    GenerateReducedModuleInterfaceAction Action;
    EXPECT_TRUE(Instance.ExecuteAction(Action));
    EXPECT_FALSE(Diags->hasErrorOccurred());
    return PCM;
  }

  llvm::SmallString<256> TestDir;
};

TEST_F(LambdaExprModulesTest, VisitsMergedLambdaBodies) {
  addFile("shared.h", R"cpp(
#pragma once
template <class T>
inline constexpr unsigned long size_of = sizeof(T);

template <class T>
inline constexpr auto probe = [] { return size_of<T>; }();
)cpp");
  addFile("mod_a.cppm", R"cpp(
module;
#include "shared.h"
export module mod_a;
export inline constexpr unsigned long kA = probe<unsigned long>;
)cpp");
  addFile("mod_b.cppm", R"cpp(
module;
#include "shared.h"
export module mod_b;
export inline constexpr unsigned long kB = probe<unsigned long>;
)cpp");

  std::string PModA = precompile("mod_a.cppm");
  std::string PModB = precompile("mod_b.cppm");
  std::string ArgA = "-fmodule-file=mod_a=" + PModA;
  std::string ArgB = "-fmodule-file=mod_b=" + PModB;

  std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
      R"cpp(
import mod_a;
import mod_b;
unsigned long use() { return kA + kB; }
)cpp",
      /*Args=*/{"-std=c++20", ArgA, ArgB});
  ASSERT_TRUE(AST);

  ImplicitLambdaBodyVisitor Visitor;
  Visitor.TraverseDecl(AST->getASTContext().getTranslationUnitDecl());

  // The TU imports two modules that each instantiate the same variable
  // template, so the closure type is merged and one call operator is left
  // without a body. The implicit traversal must still visit every lambda body
  // it encountered.
  ASSERT_GT(Visitor.NumLambdas, 0u);
  EXPECT_EQ(Visitor.NumLambdas, Visitor.NumVisitedBodies);
}

} // namespace
