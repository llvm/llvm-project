//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_XCHG_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_XCHG_H__

#include <clc/opencl/opencl-base.h>

#define __CLC_FUNCTION atomic_xchg

_CLC_OVERLOAD _CLC_DECL float __CLC_FUNCTION(volatile local float *, float);
_CLC_OVERLOAD _CLC_DECL float __CLC_FUNCTION(volatile global float *, float);
#include <clc/opencl/atomic/atomic_decl_legacy.inc>

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_XCHG_H__
