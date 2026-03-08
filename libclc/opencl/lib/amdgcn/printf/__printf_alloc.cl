//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <amdhsa_abi.h>

#define OFFSET 8

// Atomically reserves space to the printf data buffer and returns a pointer to
// it
__global char *__printf_alloc(uint bytes) {
  __constant amdhsa_implicit_kernarg_v5 *args =
      (__constant amdhsa_implicit_kernarg_v5 *)
          __builtin_amdgcn_implicitarg_ptr();
  __global char *ptr = (__global char *)args->printf_buffer;

  uint size = ((__global uint *)ptr)[1];
  uint offset = __opencl_atomic_load((__global atomic_uint *)ptr,
                                     memory_order_relaxed, memory_scope_device);

  for (;;) {
    if (OFFSET + offset + bytes > size)
      return NULL;

    if (__opencl_atomic_compare_exchange_strong(
            (__global atomic_uint *)ptr, &offset, offset + bytes,
            memory_order_relaxed, memory_order_relaxed, memory_scope_device))
      break;
  }

  return ptr + OFFSET + offset;
}
