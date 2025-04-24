//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This does not exist for fp16
#if __CLC_FPSIZE > 16

#if __CLC_FPSIZE == 32
#define STORE_HALF_BUILTIN __builtin_store_halff
#elif __CLC_FPSIZE == 64
#define STORE_HALF_BUILTIN __builtin_store_half
#else
#error "Invalid FP size"
#endif

#ifndef __CLC_SCALAR

#if __CLC_VECSIZE == 3
#define __CLC_OFFSET 4
#else
#define __CLC_OFFSET __CLC_VECSIZE
#endif

FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __private,
     STORE_HALF_BUILTIN);
FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __local,
     STORE_HALF_BUILTIN);
FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __global,
     STORE_HALF_BUILTIN);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_OFFSET, __CLC_GENTYPE, __generic,
     STORE_HALF_BUILTIN);
#endif

#undef __CLC_OFFSET
#else
FUNC(, 1, 1, __CLC_GENTYPE, __private, STORE_HALF_BUILTIN);
FUNC(, 1, 1, __CLC_GENTYPE, __local, STORE_HALF_BUILTIN);
FUNC(, 1, 1, __CLC_GENTYPE, __global, STORE_HALF_BUILTIN);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
FUNC(, 1, 1, __CLC_GENTYPE, __generic, STORE_HALF_BUILTIN);
#endif
#endif

#undef STORE_HALF_BUILTIN

#endif
