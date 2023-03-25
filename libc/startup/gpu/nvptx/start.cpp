//===-- Implementation of crt for nvptx -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc_client.h"

extern "C" int main(int argc, char **argv, char **envp);

extern "C" [[gnu::visibility("protected")]] __attribute__((nvptx_kernel)) void
_start(int argc, char **argv, char **envp, int *ret, void *in, void *out,
       void *buffer) {
  __llvm_libc::rpc::client.reset(in, out, buffer);

  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
}
