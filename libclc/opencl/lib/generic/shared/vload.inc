//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_VLOAD_NAME(x)                                                    \
  __CLC_XCONCAT(__CLC_XCONCAT(x, vload), __CLC_VECSIZE)
#define __CLC_VLOAD_HALF_NAME(x)                                               \
  __CLC_XCONCAT(__CLC_XCONCAT(x, vload_half), __CLC_VECSIZE)
#define __CLC_VLOADA_HALF_NAME(x)                                              \
  __CLC_XCONCAT(__CLC_XCONCAT(x, vloada_half), __CLC_VECSIZE)

#ifndef __CLC_SCALAR

#define __CLC_VLOAD_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)

#define __CLC_VLOAD_DEF(ADDRSPACE)                                             \
  _CLC_OVERLOAD _CLC_DEF __CLC_VLOAD_TY __CLC_VLOAD_NAME()(                    \
      size_t offset, const ADDRSPACE __CLC_SCALAR_GENTYPE *x) {                \
    return __CLC_VLOAD_NAME(__clc_)(offset, x);                                \
  }

__CLC_VLOAD_DEF(__private)
__CLC_VLOAD_DEF(__local)
__CLC_VLOAD_DEF(__constant)
__CLC_VLOAD_DEF(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VLOAD_DEF(__generic)
#endif

#undef __CLC_VLOAD_DEF
#undef __CLC_VLOAD_TY

#endif

// vload_half and vloada_half are available even if cl_khr_fp16 is unavailable.
// Declare these functions when working on float types, which we know are
// always available.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32

#define __CLC_VLOAD_HALF_DEF(ADDRSPACE)                                        \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_VLOAD_HALF_NAME()(                \
      size_t offset, const ADDRSPACE half *mem) {                              \
    return __CLC_VLOAD_HALF_NAME(__clc_)(offset, mem);                         \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_VLOADA_HALF_NAME()(               \
      size_t offset, const ADDRSPACE half *mem) {                              \
    return __CLC_VLOADA_HALF_NAME(__clc_)(offset, mem);                        \
  }

__CLC_VLOAD_HALF_DEF(__private)
__CLC_VLOAD_HALF_DEF(__local)
__CLC_VLOAD_HALF_DEF(__constant)
__CLC_VLOAD_HALF_DEF(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VLOAD_HALF_DEF(__generic)
#endif

#undef __CLC_VLOAD_HALF_DEF
#endif
#endif

#undef __CLC_VLOAD_NAME
#undef __CLC_VLOAD_HALF_NAME
#undef __CLC_VLOADA_HALF_NAME
