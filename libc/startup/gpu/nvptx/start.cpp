//===-- Implementation of crt for amdgpu ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" __attribute__((device)) int main(int argc, char **argv, char **envp);

// TODO: We shouldn't need to use the CUDA language to emit a kernel for NVPTX.
extern "C" [[gnu::visibility("protected")]] __attribute__((global)) void
_start(int argc, char **argv, char **envp, int *ret, void *in, void *out,
       void *buffer) {
  __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
}
