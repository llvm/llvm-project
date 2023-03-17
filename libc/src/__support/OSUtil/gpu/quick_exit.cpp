//===---------- GPU implementation of a quick exit function -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_GPU_QUICK_EXIT_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_GPU_QUICK_EXIT_H

#include "quick_exit.h"

#include "src/__support/RPC/rpc_client.h"
#include "src/__support/macros/properties/architectures.h"

namespace __llvm_libc {

void quick_exit(int status) {
  // TODO: Support asynchronous calls so we don't wait and exit from the GPU
  // immediately.
  rpc::client.run(
      [&](rpc::Buffer *buffer) {
        buffer->data[0] = rpc::Opcode::EXIT;
        buffer->data[1] = status;
      },
      [](rpc::Buffer *) {});

#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
  asm("exit" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  // This will terminate the entire wavefront, may not be valid with divergent
  // work items.
  asm("s_endpgm" ::: "memory");
#endif
  __builtin_unreachable();
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_GPU_QUICK_EXIT_H
