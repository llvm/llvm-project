//===-- Implementation of crt for nvptx -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"

extern "C" int main(int argc, char **argv, char **envp);

namespace LIBC_NAMESPACE {

extern "C" {
// Nvidia's 'nvlink' linker does not provide these symbols. We instead need
// to manually create them and update the globals in the loader implememtation.
uintptr_t *__init_array_start [[gnu::visibility("protected")]];
uintptr_t *__init_array_end [[gnu::visibility("protected")]];
uintptr_t *__fini_array_start [[gnu::visibility("protected")]];
uintptr_t *__fini_array_end [[gnu::visibility("protected")]];
}

using InitCallback = void(int, char **, char **);
using FiniCallback = void(void);

static void call_init_array_callbacks(int argc, char **argv, char **env) {
  size_t init_array_size = __init_array_end - __init_array_start;
  for (size_t i = 0; i < init_array_size; ++i)
    reinterpret_cast<InitCallback *>(__init_array_start[i])(argc, argv, env);
}

static void call_fini_array_callbacks() {
  size_t fini_array_size = __fini_array_end - __fini_array_start;
  for (size_t i = 0; i < fini_array_size; ++i)
    reinterpret_cast<FiniCallback *>(__fini_array_start[i])();
}

} // namespace LIBC_NAMESPACE

extern "C" [[gnu::visibility("protected"), clang::nvptx_kernel]] void
_begin(int argc, char **argv, char **env) {
  // We want the fini array callbacks to be run after other atexit
  // callbacks are run. So, we register them before running the init
  // array callbacks as they can potentially register their own atexit
  // callbacks.
  LIBC_NAMESPACE::atexit(&LIBC_NAMESPACE::call_fini_array_callbacks);
  LIBC_NAMESPACE::call_init_array_callbacks(argc, argv, env);
}

extern "C" [[gnu::visibility("protected"), clang::nvptx_kernel]] void
_start(int argc, char **argv, char **envp, int *ret) {
  // Invoke the 'main' function with every active thread that the user launched
  // the _start kernel with.
  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
}

extern "C" [[gnu::visibility("protected"), clang::nvptx_kernel]] void
_end(int retval) {
  // To finis the execution we invoke all the callbacks registered via 'atexit'
  // and then exit with the appropriate return value.
  LIBC_NAMESPACE::exit(retval);
}
