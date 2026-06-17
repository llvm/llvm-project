//===-- fakeflang.cpp - Mock Flang Driver ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Mock driver to pass CMake's compiler introspection for
/// CMake_Fortran_COMPILER. It's purpose is to not having to build the full
/// flang compiler for the runtimes-configure phase in bootstrapping-runtimes
/// builds, but only when the Fortran compiler is actually needed (e.g.
/// flang-rt-mod, libomp-mod).
///
/// To detect LLVMFlang, CMake executes
///
///   ${CMAKE_Fortran_COMPILER} -v -c -target=... CMakeFortranCompilerId.F
///
/// and expects a new file to appear in the working directory. This would
/// usually be an object file (e.g. ELF), but it doesn't matter for CMake as it
/// parses it for the preprocessor result of CMakeFortranCompilerId.F which
/// would appear as string literals in the binary file (CMake cannot execute the
/// file because it might be cross-compiling). Just passing it through the
/// preprocessor yields the same result.
///
/// The most relevant preprocessor definition is __flang__ which leads to
/// CMAKE_Fortran_COMPILER_ID="LLVMFlang".
//
//===----------------------------------------------------------------------===//

#include "flang/Version.inc"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WithColor.h"
#include <cstdint>
#include <string>

#define STRINGIFY(X) #X
#define STRINGIFY_EXPANDED(X) STRINGIFY(X)

static std::string getExecutablePath(const char *argv0) {
  void *anchor = (void *)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, anchor);
}

[[noreturn]] static void fail(llvm::Twine Error) {
  llvm::WithColor::error(llvm::errs(), "fakeflang") << Error << "\n";
  exit(EXIT_FAILURE);
}

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  std::string SelfExe = getExecutablePath(argv[0]);

  if (llvm::sys::path::stem(SelfExe) == "flang") {
    llvm::WithColor::remark(llvm::errs())
        << "This is a mock flang compiler; Use '" STRINGIFY_EXPANDED(
               CMAKE_MAKE_PROGRAM) " flang' to replace it with the real "
                                   "compiler\n";
  }

  llvm::SmallString<256> ClangExe{llvm::sys::path::parent_path(SelfExe)};
  llvm::sys::path::append(ClangExe, "clang");

  llvm::ArrayRef<const char *> AllArgs(argv, static_cast<size_t>(argc));
  bool hasDashO = AllArgs.size() > 1 &&
      llvm::any_of(AllArgs.drop_front(), [](const char *Arg) {
        return llvm::StringRef(Arg).starts_with("-o");
      });

  // Assemble invocation of the preprocessor
  // `-E`: Invoke the preprocessor
  // `-P`: No #line directives
  // `-D..`: Preprocessor definitions that CMake probes
  // `-x c`: Usually Clang would forward Fortran files to gfortran; Interpret as
  //         C for clang to preprocess the files itself
  // `-o`: -E by default emits to stdout, but CMake expects a new file to appear
  //       in the cwd
  llvm::SmallVector<llvm::StringRef, 32> Args;
  Args.append({ClangExe, "-E", "-P", "-D__flang__=1",
      "-D__flang_major__=" FLANG_VERSION_MAJOR_STRING,
      "-D__flang_minor__=" FLANG_VERSION_MINOR_STRING,
      "-D__flang_patchlevel__=" FLANG_VERSION_PATCHLEVEL_STRING, "-x", "c"});
  for (int I = 1; I < argc; ++I)
    Args.push_back(argv[I]);
  if (!hasDashO)
    Args.append({"-o", "a.out"});

  std::string ErrMsg;
  int RC = llvm::sys::ExecuteAndWait(ClangExe, Args, /*Env=*/std::nullopt,
      /*Redirects=*/{}, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);
  if (RC < 0)
    fail(ErrMsg);
  return RC;
}
