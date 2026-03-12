//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/address_space/qualifier.h>

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_OVERLOAD _CLC_DEF _CLC_CONST cl_mem_fence_flags
__clc_get_fence(const __generic void *p) {
  return __builtin_amdgcn_is_shared(p) ? CLK_LOCAL_MEM_FENCE
                                       : CLK_GLOBAL_MEM_FENCE;
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST const __global void *
__clc_to_global(const __generic void *p) {
  return __builtin_amdgcn_is_private(p) || __builtin_amdgcn_is_shared(p)
             ? NULL
             : (const __global void *)p;
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST const __local void *
__clc_to_local(const __generic void *p) {
  return __builtin_amdgcn_is_shared(p) ? (__local void *)p : NULL;
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST const __private void *
__clc_to_private(const __generic void *p) {
  return __builtin_amdgcn_is_private(p) ? (__private void *)p : NULL;
}

#endif
