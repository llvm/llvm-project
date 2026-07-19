//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clc/address_space/clc_qualifier.h"

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_DEF _CLC_OVERLOAD _CLC_CONST cl_mem_fence_flags
get_fence(__generic void *p) {
  return __clc_get_fence(p);
}

_CLC_OVERLOAD _CLC_DEF _CLC_CONST cl_mem_fence_flags
get_fence(const __generic void *p) {
  return __clc_get_fence(p);
}

_CLC_DEF _CLC_CONST __global void *__to_global(__generic void *p) {
  return __clc_to_global(p);
}

_CLC_DEF _CLC_CONST __local void *__to_local(__generic void *p) {
  return __clc_to_local(p);
}

_CLC_DEF _CLC_CONST __private void *__to_private(__generic void *p) {
  return __clc_to_private(p);
}

#endif // _CLC_GENERIC_AS_SUPPORTED
