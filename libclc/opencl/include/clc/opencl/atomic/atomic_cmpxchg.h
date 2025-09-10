//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_CMPXCHG_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_CMPXCHG_H__

#include <clc/opencl/opencl-base.h>

_CLC_OVERLOAD _CLC_DECL int atomic_cmpxchg(volatile local int *, int, int);
_CLC_OVERLOAD _CLC_DECL int atomic_cmpxchg(volatile global int *, int, int);
_CLC_OVERLOAD _CLC_DECL uint atomic_cmpxchg(volatile local uint *, uint, uint);
_CLC_OVERLOAD _CLC_DECL uint atomic_cmpxchg(volatile global uint *, uint, uint);

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_CMPXCHG_H__
