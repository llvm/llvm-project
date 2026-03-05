//===-- amdhsa_abi.h - AMDHSA ABI definition utilities --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __AMDHSA_ABI_H
#define __AMDHSA_ABI_H

#include <stddef.h>
#include <stdint.h>

typedef struct __attribute__((aligned(8))) amdhsa_implicit_kernarg_v5 {
  uint32_t block_count[3];
  uint16_t group_size[3];
  uint16_t remainder[3];
  char reserved0[16];
  uint64_t global_offset[3];
  uint16_t grid_dims;
  char reserved1[14];
  __attribute__((opencl_global)) void *hostcall_buffer;
  __attribute__((opencl_global)) void *multigrid_sync_arg;
  __attribute__((opencl_global)) void *heap_v1;
  __attribute__((opencl_global)) void *default_queue;
  __attribute__((opencl_global)) void *completion_action;
  char reserved2[72];
  uint32_t private_base; // Unused on gfx9+
  uint32_t shared_base;  // Unused on gfx9+
  __attribute__((opencl_global)) void *queue_ptr;
  char reserved3[48];
} amdhsa_implicit_kernarg_v5;

_Static_assert(sizeof(amdhsa_implicit_kernarg_v5) == 256, "wrong struct size");

_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, block_count[0]) == 0,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, block_count[1]) == 4,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, block_count[2]) == 8,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, group_size[0]) == 12,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, group_size[1]) == 14,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, group_size[2]) == 16,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, remainder[0]) == 18,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, remainder[1]) == 20,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, remainder[2]) == 22,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, global_offset[0]) == 40,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, global_offset[1]) == 48,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, global_offset[2]) == 56,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, grid_dims) == 64,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, hostcall_buffer) == 80,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, multigrid_sync_arg) == 88,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, heap_v1) == 96,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, default_queue) == 104,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, completion_action) == 112,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, private_base) == 192,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, shared_base) == 196,
               "wrong offset");
_Static_assert(offsetof(amdhsa_implicit_kernarg_v5, queue_ptr) == 200,
               "wrong offset");

#endif // __AMDHSA_ABI_H
