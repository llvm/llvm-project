//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcfunc.h>
#include <clc/clctypes.h>

#ifdef cl_khr_global_int32_base_atomics
_CLC_OVERLOAD _CLC_DECL int atom_inc(volatile global int *p);
_CLC_OVERLOAD _CLC_DECL unsigned int atom_inc(volatile global unsigned int *p);
#endif // cl_khr_global_int32_base_atomics

#ifdef cl_khr_local_int32_base_atomics
_CLC_OVERLOAD _CLC_DECL int atom_inc(volatile local int *p);
_CLC_OVERLOAD _CLC_DECL unsigned int atom_inc(volatile local unsigned int *p);
#endif // cl_khr_local_int32_base_atomics

#ifdef cl_khr_int64_base_atomics
_CLC_OVERLOAD _CLC_DECL long atom_inc(volatile global long *p);
_CLC_OVERLOAD _CLC_DECL unsigned long
atom_inc(volatile global unsigned long *p);
_CLC_OVERLOAD _CLC_DECL long atom_inc(volatile local long *p);
_CLC_OVERLOAD _CLC_DECL unsigned long atom_inc(volatile local unsigned long *p);
#endif // cl_khr_int64_base_atomics
