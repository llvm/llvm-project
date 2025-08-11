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
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Interpreter/RemoteJITUtils.h"
#include "clang/Interpreter/Value.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"

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
  llvm::SmallString<256> RuntimePath(llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&getOrcRuntimePath)));
  removePathComponent(5, RuntimePath);
  llvm::sys::path::append(RuntimePath, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                          CLANG_VERSION_MAJOR_STRING, "lib");

  llvm::Triple SystemTriple(llvm::sys::getProcessTriple());
  if (SystemTriple.isOSBinFormatMachO()) {
    llvm::sys::path::append(RuntimePath, "darwin", "liborc_rt_osx.a");
  } else if (SystemTriple.isOSBinFormatELF()) {
    llvm::sys::path::append(RuntimePath, "x86_64-unknown-linux-gnu",
                            "liborc_rt.a");
  }

  return RuntimePath.str().str();
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

  std::unique_ptr<llvm::orc::LLJITBuilder> JB;

  llvm::Triple SystemTriple(llvm::sys::getProcessTriple());

  if ((SystemTriple.isOSBinFormatELF() || SystemTriple.isOSBinFormatMachO())) {
    std::string OOPExecutor = getExecutorPath();
    std::string OrcRuntimePath = getOrcRuntimePath();
    bool UseSharedMemory = false;
    std::string SlabAllocateSizeString = "";
    std::unique_ptr<llvm::orc::ExecutorProcessControl> EPC;
    EPC = ExitOnError(launchExecutor(OOPExecutor, UseSharedMemory,
                                     SlabAllocateSizeString,
                                     [=] { // Lambda defined inline
                                       auto redirect = [](int from, int to) {
                                         if (from != to) {
                                           dup2(from, to);
                                           close(from);
                                         }
                                       };

                                       redirect(0, STDIN_FILENO);
                                       redirect(1, STDOUT_FILENO);
                                       redirect(2, STDERR_FILENO);

                                       setvbuf(stdout, nullptr, _IONBF, 0);
                                       setvbuf(stderr, nullptr, _IONBF, 0);
                                     }));
    if (EPC) {
      CB.SetTargetTriple(EPC->getTargetTriple().getTriple());
      JB = ExitOnError(clang::Interpreter::createLLJITBuilder(std::move(EPC),
                                                              OrcRuntimePath));
    }
  }

  return cantFail(clang::Interpreter::create(std::move(CI), std::move(JB)));
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
