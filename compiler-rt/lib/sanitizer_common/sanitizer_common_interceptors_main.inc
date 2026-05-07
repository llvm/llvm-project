//===-- sanitizer_common_interceptors_main.inc -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Opt-in Linux/glibc main wrapper via __libc_start_main interception.
//
// A sanitizer enables this file by defining:
//   #define MAIN_WRAPPER(main, argc, argv, envp) \
//     ToolMainWrapper(main, argc, argv, envp)
// before including it. `main` is the real user main (__sanitizer::MainFnTy);
// argc, argv, and envp are the startup arguments forwarded to the wrapper.
// The wrapper returns the process exit code and decides when to call main.
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "sanitizer_common.h"
#include "sanitizer_platform.h"

// __libc_start_main is a dynamic symbol in glibc.
#if SANITIZER_LINUX && SANITIZER_GLIBC && defined(MAIN_WRAPPER)
namespace __sanitizer {
namespace {

MainFnTy real_main;

int WrappedMain(int argc, char **argv, char **envp) {
  return MAIN_WRAPPER(real_main, argc, argv, envp);
}

}  // namespace
}  // namespace __sanitizer

INTERCEPTOR(int, __libc_start_main, __sanitizer::MainFnTy main, int argc,
            char **argv, void (*init)(), void (*fini)(), void (*rtld_fini)(),
            void *stack_end) {
  // Lazily obtain the address of real __libc_start_main.
  // Equivalent to the single-threaded version of functionl local static
  // initialization.
  // This is thread-safe as __libc_start_main is expected to be called AT FIRST
  // from _start (the entry point of ELF), which is single-threaded.
  if (!REAL(__libc_start_main))
    CHECK(INTERCEPT_FUNCTION(__libc_start_main));
  __sanitizer::real_main = main;
  return REAL(__libc_start_main)(__sanitizer::WrappedMain, argc, argv, init,
                                 fini, rtld_fini, stack_end);
}
#endif
