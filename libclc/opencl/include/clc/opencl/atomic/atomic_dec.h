//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOMIC_DEC_H__
#define __CLC_OPENCL_ATOMIC_ATOMIC_DEC_H__

#include <clc/opencl/opencl-base.h>

_CLC_OVERLOAD _CLC_DECL int atomic_dec(volatile local int *);
_CLC_OVERLOAD _CLC_DECL int atomic_dec(volatile global int *);
_CLC_OVERLOAD _CLC_DECL uint atomic_dec(volatile local uint *);
_CLC_OVERLOAD _CLC_DECL uint atomic_dec(volatile global uint *);

#endif // __CLC_OPENCL_ATOMIC_ATOMIC_DEC_H__
