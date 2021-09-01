//===- unittests/Interpreter/InterpreterExceptionTest.cpp -----------------===//
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

#include "clang/Interpreter/Interpreter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ManagedStatic.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {
using Args = std::vector<const char *>;
static std::unique_ptr<Interpreter>
createInterpreter(const Args &ExtraArgs = {},
                  DiagnosticConsumer *Client = nullptr) {
  Args ClangArgs = {"-Xclang", "-emit-llvm-only"};
  ClangArgs.insert(ClangArgs.end(), ExtraArgs.begin(), ExtraArgs.end());
  auto CI = cantFail(clang::IncrementalCompilerBuilder::create(ClangArgs));
  if (Client)
    CI->getDiagnostics().setClient(Client, /*ShouldOwnClient=*/false);
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

// This function isn't referenced outside its translation unit, but it
// can't use the "static" keyword because its address is used for
// GetMainExecutable (since some platforms don't support taking the
// address of main, and some platforms can't implement GetMainExecutable
// without being given the address of a function in the main executable).
std::string GetExecutablePath(const char *Argv0, void *MainAddr) {
  return llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
}

static std::string MakeResourcesPath() {
  // Dir is bin/ or lib/, depending on where BinaryPath is.
  void *MainAddr = (void *)(intptr_t)GetExecutablePath;
  std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr, MainAddr);

  // build/tools/clang/unittests/Interpreter/Executable -> build/
  llvm::StringRef Dir = llvm::sys::path::parent_path(BinaryPath);

  Dir = llvm::sys::path::parent_path(Dir);
  Dir = llvm::sys::path::parent_path(Dir);
  Dir = llvm::sys::path::parent_path(Dir);
  Dir = llvm::sys::path::parent_path(Dir);
  Dir = llvm::sys::path::parent_path(Dir);
  SmallString<128> P(Dir);
  llvm::sys::path::append(P, Twine("lib") + CLANG_LIBDIR_SUFFIX, "clang",
                          CLANG_VERSION_STRING);

  return std::string(P.str());
}

TEST(InterpreterTest, CatchException) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  {
    auto J = llvm::orc::LLJITBuilder().create();
    if (!J) {
      // The platform does not support JITs.
      // We can't use llvm::consumeError as it needs typeinfo for ErrorInfoBase.
      auto E = J.takeError();
      (void)E;
      return;
    }
  }
  const char ExceptionCode[] =
      R"(
#include <stdexcept>
#include <stdio.h>

static void ThrowerAnError(const char* Name) {
  throw std::runtime_error(Name);
}

extern "C" int throw_exception() {
  try {
    ThrowerAnError("In JIT");
  } catch (const std::exception& E) {
    printf("Caught: '%s'\n", E.what());
  } catch (...) {
    printf("Unknown exception\n");
  }
  ThrowerAnError("From JIT");
  return 0;
}
    )";
  std::string ResourceDir = MakeResourcesPath();
  std::unique_ptr<Interpreter> Interp =
      createInterpreter({"-resource-dir", ResourceDir.c_str()});
  // FIXME: Re-enable the excluded target triples.
  const clang::CompilerInstance *CI = Interp->getCompilerInstance();
  const llvm::Triple &Triple = CI->getASTContext().getTargetInfo().getTriple();
  // FIXME: PPC fails due to `Symbols not found: [DW.ref.__gxx_personality_v0]`
  // The current understanding is that the JIT should emit this symbol if it was
  // not (eg. the way passing clang -fPIC does it).
  if (Triple.isPPC())
    return;

  // FIXME: ARM fails due to `Not implemented relocation type!`
  if (Triple.isARM())
    return;

  // FIXME: Hexagon fails due to `No available targets are compatible with
  // triple "x86_64-unknown-linux-gnu"`
  if (Triple.getArch() == llvm::Triple::hexagon)
    return;

  // Adjust the resource-dir
  llvm::cantFail(Interp->ParseAndExecute(ExceptionCode));
  testing::internal::CaptureStdout();
  auto ThrowException =
      (int (*)())llvm::cantFail(Interp->getSymbolAddress("throw_exception"));
  EXPECT_THROW(ThrowException(), std::exception);
  std::string CapturedStdOut = testing::internal::GetCapturedStdout();
  EXPECT_EQ(CapturedStdOut, "Caught: 'In JIT'\n");

  llvm::llvm_shutdown();
}

} // end anonymous namespace
