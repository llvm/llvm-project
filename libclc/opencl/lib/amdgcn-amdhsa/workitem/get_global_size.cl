//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/opencl/opencl-base.h>

_CLC_DEF _CLC_OVERLOAD size_t get_global_size(uint dim) {
  __constant uint *ptr = (__constant uint *)__builtin_amdgcn_dispatch_ptr();
  if (dim < 3)
    return ptr[3 + dim];
  return 1;
}
