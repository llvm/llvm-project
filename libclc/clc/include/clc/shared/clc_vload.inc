//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_VLOAD_NAME __CLC_XCONCAT(__clc_vload, __CLC_VECSIZE)
#define __CLC_VLOAD_HALF_NAME __CLC_XCONCAT(__clc_vload_half, __CLC_VECSIZE)
#define __CLC_VLOADA_HALF_NAME __CLC_XCONCAT(__clc_vloada_half, __CLC_VECSIZE)

#ifndef __CLC_SCALAR

#define __CLC_VLOAD_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)

#define __CLC_VLOAD_DECL(ADDRSPACE)                                            \
  _CLC_OVERLOAD _CLC_DECL __CLC_VLOAD_TY __CLC_VLOAD_NAME(                     \
      size_t offset, const ADDRSPACE __CLC_SCALAR_GENTYPE *x);

__CLC_VLOAD_DECL(__private)
__CLC_VLOAD_DECL(__local)
__CLC_VLOAD_DECL(__constant)
__CLC_VLOAD_DECL(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VLOAD_DECL(__generic)
#endif

#undef __CLC_VLOAD_DECL
#undef __CLC_VLOAD_TY

#endif // __CLC_SCALAR

// vload_half and vloada_half are available even if cl_khr_fp16 is unavailable.
// Declare these functions when working on float types, which we know are
// always available.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32

#define __CLC_VLOAD_HALF_DECL(ADDRSPACE)                                       \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_VLOAD_HALF_NAME(                 \
      size_t offset, const ADDRSPACE half *mem);                               \
                                                                               \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE __CLC_VLOADA_HALF_NAME(                \
      size_t offset, const ADDRSPACE half *mem);

__CLC_VLOAD_HALF_DECL(__private)
__CLC_VLOAD_HALF_DECL(__local)
__CLC_VLOAD_HALF_DECL(__constant)
__CLC_VLOAD_HALF_DECL(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VLOAD_HALF_DECL(__generic)
#endif

#undef __CLC_VLOAD_HALF_DECL

#endif
#endif

#undef __CLC_VLOAD_NAME
#undef __CLC_VLOAD_HALF_NAME
#undef __CLC_VLOADA_HALF_NAME
