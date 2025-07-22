//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef cl_khr_global_int32_extended_atomics
#define __CLC_FUNCTION atom_xor
#define __CLC_ADDRESS_SPACE global
#include <clc/atomic/atom_decl_int32.inc>
#endif // cl_khr_global_int32_extended_atomics

#ifdef cl_khr_local_int32_extended_atomics
#define __CLC_FUNCTION atom_xor
#define __CLC_ADDRESS_SPACE local
#include <clc/atomic/atom_decl_int32.inc>
#endif // cl_khr_local_int32_extended_atomics

#ifdef cl_khr_int64_extended_atomics
#define __CLC_FUNCTION atom_xor
#include <clc/atomic/atom_decl_int64.inc>
#endif // cl_khr_int64_extended_atomics
