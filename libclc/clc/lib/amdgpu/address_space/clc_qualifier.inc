//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_CLC_OVERLOAD _CLC_DEF _CLC_CONST cl_mem_fence_flags
__clc_get_fence(__CLC_CONST_TYPE __generic void *p) {
  return __builtin_amdgcn_is_shared(p) ? CLK_LOCAL_MEM_FENCE
                                       : CLK_GLOBAL_MEM_FENCE;
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_CONST_TYPE __global void *
__clc_to_global(__CLC_CONST_TYPE __generic void *p) {
  return __builtin_amdgcn_is_private(p) || __builtin_amdgcn_is_shared(p)
             ? NULL
             : (__CLC_CONST_TYPE __global void *)p;
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_CONST_TYPE __local void *
__clc_to_local(__CLC_CONST_TYPE __generic void *p) {
  return __builtin_amdgcn_is_shared(p) ? (__CLC_CONST_TYPE __local void *)p
                                       : NULL;
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __CLC_CONST_TYPE __private void *
__clc_to_private(__CLC_CONST_TYPE __generic void *p) {
  return __builtin_amdgcn_is_private(p) ? (__CLC_CONST_TYPE __private void *)p
                                        : NULL;
}
