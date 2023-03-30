//===-- Shared memory RPC client / server utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_RPC_RPC_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_RPC_RPC_UTILS_H

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"

namespace __llvm_libc {
namespace rpc {

/// Suspend the thread briefly to assist the thread scheduler during busy loops.
LIBC_INLINE void sleep_briefly() {
#if defined(LIBC_TARGET_ARCH_IS_NVPTX) && __CUDA_ARCH__ >= 700
  asm("nanosleep.u32 64;" ::: "memory");
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  __builtin_amdgcn_s_sleep(2);
#else
  // Simply do nothing if sleeping isn't supported on this platform.
#endif
}

} // namespace rpc
} // namespace __llvm_libc

#endif
