//===-- Implementation of crt for gpu -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/gpu/app.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/RPC/rpc_client.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"

extern "C" int main(int argc, char **argv, char **envp);
extern "C" void __cxa_finalize(void *dso);

namespace LIBC_NAMESPACE_DECL {

DataEnvironment app;

} // namespace LIBC_NAMESPACE_DECL

extern "C" [[gnu::visibility("protected"), clang::device_kernel]]
void _begin(int, char **, char **env) {
  // The LLVM offloading runtime will automatically call any present global
  // constructors and destructors so we defer that handling.
  __atomic_store_n(&LIBC_NAMESPACE::app.env_ptr,
                   reinterpret_cast<uintptr_t *>(env), __ATOMIC_RELAXED);
}

extern "C" [[gnu::visibility("protected"), clang::device_kernel]] void
_start(int argc, char **argv, char **envp, int *ret) {
  // Invoke the 'main' function with every active thread that the user launched
  // the _start kernel with.
  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
}

extern "C" [[gnu::visibility("protected"), clang::device_kernel]]
void _end() {
  // Only a single thread should call the destructors registred with 'atexit'.
  // The loader utility will handle the actual exit and return code cleanly.
  __cxa_finalize(nullptr);
}
