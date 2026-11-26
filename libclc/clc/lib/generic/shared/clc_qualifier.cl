//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/address_space/clc_qualifier.h"

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_OVERLOAD _CLC_DEF _CLC_CONST cl_mem_fence_flags
__clc_get_fence(__generic void *p) {
  return __clc_get_fence((const __generic void *)p);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __global void *
__clc_to_global(__generic void *p) {
  return (__global void *)__clc_to_global((const __generic void *)p);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __local void *
__clc_to_local(__generic void *p) {
  return (__local void *)__clc_to_local((const __generic void *)p);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST __private void *
__clc_to_private(__generic void *p) {
  return (__private void *)__clc_to_private((const __generic void *)p);
}

#endif
