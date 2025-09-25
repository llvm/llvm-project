//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_ATOMIC_ATOM_MIN_H__
#define __CLC_OPENCL_ATOMIC_ATOM_MIN_H__

#include <clc/opencl/opencl-base.h>

#ifdef cl_khr_global_int32_extended_atomics
#define __CLC_FUNCTION atom_min
#define __CLC_ADDRESS_SPACE global
#include <clc/opencl/atomic/atom_decl_int32.inc>
#endif // cl_khr_global_int32_extended_atomics

#ifdef cl_khr_local_int32_extended_atomics
#define __CLC_FUNCTION atom_min
#define __CLC_ADDRESS_SPACE local
#include <clc/opencl/atomic/atom_decl_int32.inc>
#endif // cl_khr_local_int32_extended_atomics

#ifdef cl_khr_int64_extended_atomics
#define __CLC_FUNCTION atom_min
#include <clc/opencl/atomic/atom_decl_int64.inc>
#endif // cl_khr_int64_extended_atomics

#endif // __CLC_OPENCL_ATOMIC_ATOM_MIN_H__
