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
///
/// Whether a particular invocation is CMake's compiler detection is detected
/// via the filename or when the FLANG_BOOTSTRAP_PROBE environment variable is
/// set. Any other invocation is forwarded, unmodified, to the real flang
/// driver that must live right next to this program.
//
//===----------------------------------------------------------------------===//

#include "flang/Version.inc"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WithColor.h"
#include <cstdint>
#include <string>

static std::string getExecutablePath(const char *argv0) {
  void *anchor = (void *)(intptr_t)getExecutablePath;
  return llvm::sys::fs::getMainExecutable(argv0, anchor);
}

[[noreturn]] static void fail(llvm::Twine Error) {
  llvm::WithColor::error(llvm::errs(), "fakeflang") << Error << "\n";
  exit(EXIT_FAILURE);
}

/// A CMake compiler detection probe is assumed if
///
/// 1. FLANG_BOOTSTRAP_PROBE=1 has been set during runtimes-configure
///    (requires CMake 4.2+), or
///
/// 2. the file being compiled is CMakeFortranCompilerId.F; all supported
///    CMake versions below 4.2 use this filename.
static bool isCompilerIdProbe(llvm::ArrayRef<const char *> OrigArgs) {
  if (llvm::sys::Process::GetEnv("FLANG_BOOTSTRAP_PROBE"))
    return true;

  return llvm::any_of(OrigArgs, [](const char *Arg) {
    llvm::StringRef fname = llvm::sys::path::filename(Arg);
    return fname == "CMakeFortranCompilerId.F" || fname == "CMakeTestGNU.c";
  });
}

/// Forward the invocation directly to Flang. Only reached for invocations
/// that are not CMake's compiler-id probe, i.e. after the caller's build has
/// already established (through a real dependency, not this mock) that
/// flang is up to date.
static int main_flang(
    llvm::StringRef SelfExe, llvm::ArrayRef<const char *> OrigArgs) {
  llvm::SmallString<256> FlangExe{llvm::sys::path::parent_path(SelfExe)};
  llvm::sys::path::append(FlangExe, "flang");

  llvm::SmallVector<llvm::StringRef> Args;
  Args.push_back(FlangExe);
  llvm::append_range(Args, OrigArgs);

  std::string ErrMsg;
  int RC = llvm::sys::ExecuteAndWait(FlangExe, Args, /*Env=*/std::nullopt,
      /*Redirects=*/{}, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);
  if (RC < 0)
    fail(ErrMsg);
  return RC;
}

/// Use Clang to emulate Flang's preprocessor, answering CMake's
/// compiler-id probe without needing the real flang to be ready yet.
static int main_probe(
    llvm::StringRef SelfExe, llvm::ArrayRef<const char *> OrigArgs) {
  llvm::WithColor::warning(llvm::errs(), "fakeflang")
      << "Emulating flang for compiler probing\nFor anything other than "
         "bootstrapping the Flang toolchain, just use 'flang' instead\n";

  llvm::SmallString<256> ClangExe{llvm::sys::path::parent_path(SelfExe)};
  llvm::sys::path::append(ClangExe, "clang");

  bool hasDashO = llvm::any_of(OrigArgs,
      [](const char *Arg) { return llvm::StringRef(Arg).starts_with("-o"); });

  // Assemble invocation of the preprocessor
  // `-E`: Invoke the preprocessor
  // `-P`: No #line directives
  // `-D..`: Preprocessor definitions that CMake probes
  // `-D__fakeflang_probe__=1`: To identify fakeflang probe mode
  //                            (for regression tests)
  // `-fgnuc-version=0`: Prevent clang from implicitly defining __GNUC__;
  //                     CMake's CMakeFortranCompilerId.F.in checks
  //                     __GNUC__before __flang__, so without this flag the
  //                     compiler is misidentified as GNU.
  // `-x c`: Usually Clang would forward Fortran files to gfortran; Interpret as
  //         C for clang to preprocess the files itself
  // `-o`: -E by default emits to stdout, but CMake expects a new file to appear
  //       in the cwd
  llvm::SmallVector<llvm::StringRef, 32> Args;
  Args.append({ClangExe, "-E", "-P", "-D__flang__=1",
      "-D__flang_major__=" FLANG_VERSION_MAJOR_STRING,
      "-D__flang_minor__=" FLANG_VERSION_MINOR_STRING,
      "-D__flang_patchlevel__=" FLANG_VERSION_PATCHLEVEL_STRING,
      "-D__fakeflang_probe__=1", "-fgnuc-version=0", "-x", "c"});
  for (llvm::StringRef arg : OrigArgs) {
    // Sometime CMake tries to invoke the preprocessor
    if (arg == "-cpp" || arg == "-E")
      continue;

    Args.push_back(arg);
  }
  if (!hasDashO)
    Args.append({"-o", "a.out"});

  std::string ErrMsg;
  int RC = llvm::sys::ExecuteAndWait(ClangExe, Args, /*Env=*/std::nullopt,
      /*Redirects=*/{}, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);
  if (RC < 0)
    fail(ErrMsg);
  return RC;
}

int main(int argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);
  std::string SelfExe = getExecutablePath(argv[0]);
  llvm::ArrayRef<const char *> OrigArgs(argv, static_cast<size_t>(argc));
  OrigArgs = OrigArgs.drop_front();

  if (isCompilerIdProbe(OrigArgs))
    return main_probe(SelfExe, OrigArgs);

  return main_flang(SelfExe, OrigArgs);
}
