//===- unittests/Interpreter/OutOfProcessInterpreterTest.cpp --- Interpreter
// tests when Out-of-Process ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Clang's Interpreter library.
//
//===----------------------------------------------------------------------===//

#include "InterpreterTestFixture.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/OutOfProcessJITConfig.h"
#include "clang/Interpreter/RemoteJITUtils.h"
#include "clang/Interpreter/Value.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

llvm::ExitOnError ExitOnError;

#ifdef _WIN32
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif

namespace {

using Args = std::vector<const char *>;

static void removePathComponent(unsigned N, llvm::SmallString<256> &Path) {
  for (unsigned i = 0; i < N; ++i)
    llvm::sys::path::remove_filename(Path);
}

static std::string getExecutorPath() {
  llvm::SmallString<256> ExecutorPath(llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getExecutorPath)));
  removePathComponent(5, ExecutorPath);
  llvm::sys::path::append(ExecutorPath, "bin", "llvm-jitlink-executor");
  return ExecutorPath.str().str();
}

static std::string getOrcRuntimePath() {
  clang::DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());

  clang::IgnoringDiagConsumer DiagConsumer;
  clang::DiagnosticsEngine Diags(DiagID, DiagOpts, &DiagConsumer, false);
  std::vector<const char *> Args = {"clang", "--version"};
  clang::driver::Driver D("clang", llvm::sys::getProcessTriple(), Diags);
  D.setCheckInputsExist(false);

  std::unique_ptr<clang::driver::Compilation> C(D.BuildCompilation(Args));
  if (!C) {
    return "";
  }

  const clang::driver::ToolChain &TC = C->getDefaultToolChain();
  std::optional<std::string> CompilerRTPath = TC.getCompilerRTPath();

  // Fallback: Calculate base path similar to clang-repl logic.
  llvm::SmallString<256> BasePath(llvm::sys::fs::getMainExecutable(
      "clang", reinterpret_cast<void *>(&getOrcRuntimePath)));
  removePathComponent(5, BasePath);

  // Append libdir, clang, and version.

  llvm::sys::path::append(BasePath, *CompilerRTPath);

  std::cout << "BasePath: " << BasePath.str().str() << "\n";

  if (llvm::sys::fs::exists(BasePath.str().str() + "/liborc_rt_osx.a")) {
    return BasePath.str().str() + "/liborc_rt_osx.a";
  } else if (llvm::sys::fs::exists(BasePath.str().str() + "/liborc_rt.a")) {
    return BasePath.str().str() + "/liborc_rt.a";
  } else if (llvm::sys::fs::exists(BasePath.str().str() +
                                   "/liborc_rt-x86_64.a")) {
    return BasePath.str().str() + "liborc_rt-x86_64.a";
  } else
    return "";
}

static std::unique_ptr<Interpreter>
createInterpreterWithRemoteExecution(const Args &ExtraArgs = {},
                                     DiagnosticConsumer *Client = nullptr) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  llvm::append_range(ClangArgs, ExtraArgs);
  auto CB = clang::IncrementalCompilerBuilder();
  CB.SetCompilerArgs(ClangArgs);
  auto CI = cantFail(CB.CreateCpp());
  if (Client)
    CI->getDiagnostics().setClient(Client, /*ShouldOwnClient=*/false);

  OutOfProcessJITConfig OutOfProcessConfig;
  OutOfProcessConfig.OOPExecutor = getExecutorPath();
  OutOfProcessConfig.UseSharedMemory = false;
  OutOfProcessConfig.SlabAllocateSizeString = "";
  OutOfProcessConfig.IsOutOfProcess = true;
  OutOfProcessConfig.OrcRuntimePath = getOrcRuntimePath();


  std::cout << "OrcRuntimePath: " << OutOfProcessConfig.OrcRuntimePath << "\n";

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;

  return cantFail(
      clang::Interpreter::create(std::move(CI), OutOfProcessConfig));
}

static size_t DeclsSize(TranslationUnitDecl *PTUDecl) {
  return std::distance(PTUDecl->decls().begin(), PTUDecl->decls().end());
}

TEST_F(InterpreterTestBase, SanityWithRemoteExecution) {
  if (!HostSupportsJIT())
    GTEST_SKIP();

  std::unique_ptr<Interpreter> Interp = createInterpreterWithRemoteExecution();

  using PTU = PartialTranslationUnit;

  PTU &R1(cantFail(Interp->Parse("void g(); void g() {}")));
  EXPECT_EQ(2U, DeclsSize(R1.TUPart));

  PTU &R2(cantFail(Interp->Parse("int i;")));
  EXPECT_EQ(1U, DeclsSize(R2.TUPart));
}

} // end anonymous namespace
