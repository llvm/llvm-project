//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This does exist only for fp32
#if __CLC_FPSIZE == 32
#ifndef __CLC_SCALAR

__CLC_FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __private);
__CLC_FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __local);
__CLC_FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __global);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_FUNC(__CLC_VECSIZE, __CLC_VECSIZE, __CLC_GENTYPE, __generic);
#endif

#undef __CLC_OFFSET
#else
__CLC_FUNC(, 1, __CLC_GENTYPE, __private);
__CLC_FUNC(, 1, __CLC_GENTYPE, __local);
__CLC_FUNC(, 1, __CLC_GENTYPE, __global);
#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_FUNC(, 1, __CLC_GENTYPE, __generic);
#endif
#endif
#endif
