//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_ADDRESS_SPACE_CLC_QUALIFIER_H__
#define __CLC_ADDRESS_SPACE_CLC_QUALIFIER_H__

#include "clc/clcfunc.h"

#if _CLC_GENERIC_AS_SUPPORTED

_CLC_OVERLOAD _CLC_DECL _CLC_CONST cl_mem_fence_flags __clc_get_fence(void *p);
_CLC_OVERLOAD _CLC_DECL _CLC_CONST cl_mem_fence_flags
__clc_get_fence(const void *p);

_CLC_OVERLOAD _CLC_DECL _CLC_CONST __global void *__clc_to_global(void *p);
_CLC_OVERLOAD _CLC_DECL _CLC_CONST const __global void *
__clc_to_global(const void *p);

_CLC_OVERLOAD _CLC_DECL _CLC_CONST __local void *__clc_to_local(void *p);
_CLC_OVERLOAD _CLC_DECL _CLC_CONST const __local void *
__clc_to_local(const void *p);

_CLC_OVERLOAD _CLC_DECL _CLC_CONST __private void *__clc_to_private(void *p);
_CLC_OVERLOAD _CLC_DECL _CLC_CONST const __private void *
__clc_to_private(const void *p);
#endif // _CLC_GENERIC_AS_SUPPORTED

#endif // __CLC_ADDRESS_SPACE_CLC_QUALIFIER_H__
