//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define CLC_VLOAD_NAME __CLC_XCONCAT(__clc_vload, __CLC_VECSIZE)
#define CLC_VLOAD_HALF_NAME __CLC_XCONCAT(__clc_vload_half, __CLC_VECSIZE)
#define CLC_VLOADA_HALF_NAME __CLC_XCONCAT(__clc_vloada_half, __CLC_VECSIZE)

#ifndef __CLC_SCALAR

#define CLC_VLOAD_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)

#define CLC_VLOAD_DECL(ADDRSPACE)                                              \
  _CLC_OVERLOAD _CLC_DECL CLC_VLOAD_TY CLC_VLOAD_NAME(                         \
      size_t offset, const ADDRSPACE __CLC_SCALAR_GENTYPE *x);

CLC_VLOAD_DECL(__private)
CLC_VLOAD_DECL(__local)
CLC_VLOAD_DECL(__constant)
CLC_VLOAD_DECL(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
CLC_VLOAD_DECL(__generic)
#endif

#undef CLC_VLOAD_DECL
#undef CLC_VLOAD_TY

#endif // __CLC_SCALAR

// vload_half and vloada_half are available even if cl_khr_fp16 is unavailable.
// Declare these functions when working on float types, which we know are
// always available.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32

#define CLC_VLOAD_HALF_DECL(ADDRSPACE)                                         \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE CLC_VLOAD_HALF_NAME(                   \
      size_t offset, const ADDRSPACE half *mem);                               \
                                                                               \
  _CLC_OVERLOAD _CLC_DECL __CLC_GENTYPE CLC_VLOADA_HALF_NAME(                  \
      size_t offset, const ADDRSPACE half *mem);

CLC_VLOAD_HALF_DECL(__private)
CLC_VLOAD_HALF_DECL(__local)
CLC_VLOAD_HALF_DECL(__constant)
CLC_VLOAD_HALF_DECL(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
CLC_VLOAD_HALF_DECL(__generic)
#endif

#undef CLC_VLOAD_HALF_DECL

#endif
#endif

#undef CLC_VLOAD_NAME
#undef CLC_VLOAD_HALF_NAME
#undef CLC_VLOADA_HALF_NAME
