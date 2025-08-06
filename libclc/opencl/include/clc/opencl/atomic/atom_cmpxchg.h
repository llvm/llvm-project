//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOM_CMPXCHG_H__
#define __CLC_OPENCL_ATOMIC_ATOM_CMPXCHG_H__

#include <clc/opencl/opencl-base.h>

#include <clc/clcfunc.h>
#include <clc/clctypes.h>

#ifdef cl_khr_global_int32_base_atomics
_CLC_OVERLOAD _CLC_DECL int atom_cmpxchg(volatile global int *p, int cmp,
                                         int val);
_CLC_OVERLOAD _CLC_DECL unsigned int
atom_cmpxchg(volatile global unsigned int *p, unsigned int cmp,
             unsigned int val);
#endif // cl_khr_global_int32_base_atomics

#ifdef cl_khr_local_int32_base_atomics
_CLC_OVERLOAD _CLC_DECL int atom_cmpxchg(volatile local int *p, int cmp,
                                         int val);
_CLC_OVERLOAD _CLC_DECL unsigned int
atom_cmpxchg(volatile local unsigned int *p, unsigned int cmp,
             unsigned int val);
#endif // cl_khr_local_int32_base_atomics

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DECL long atom_cmpxchg(volatile global long *p, long cmp,
                                          long val);
_CLC_OVERLOAD _CLC_DECL unsigned long
atom_cmpxchg(volatile global unsigned long *p, unsigned long cmp,
             unsigned long val);
_CLC_OVERLOAD _CLC_DECL long atom_cmpxchg(volatile local long *p, long cmp,
                                          long val);
_CLC_OVERLOAD _CLC_DECL unsigned long
atom_cmpxchg(volatile local unsigned long *p, unsigned long cmp,
             unsigned long val);
#endif // cl_khr_int64_base_atomics

#endif // __CLC_OPENCL_ATOMIC_ATOM_CMPXCHG_H__
