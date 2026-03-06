//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <amdhsa_abi.h>
#include <clc/opencl/opencl-base.h>

_CLC_DEF _CLC_OVERLOAD size_t get_global_size(uint dim) {
  if (dim > 2)
    return 1;
  __constant amdhsa_implicit_kernarg_v5 *args =
      (__constant amdhsa_implicit_kernarg_v5 *)
          __builtin_amdgcn_implicitarg_ptr();
  return args->block_count[dim] * (uint)args->group_size[dim] +
         (uint)args->remainder[dim];
}
