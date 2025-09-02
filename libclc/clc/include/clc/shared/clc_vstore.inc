//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_VSTORE_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)
#define __CLC_VSTORE_NAME __CLC_XCONCAT(__clc_vstore, __CLC_VECSIZE)
#define __CLC_VSTORE_HALF_NAME(x)                                              \
  __CLC_XCONCAT(__CLC_XCONCAT(__clc_vstore_half, __CLC_VECSIZE), x)
#define __CLC_VSTOREA_HALF_NAME(x)                                             \
  __CLC_XCONCAT(__CLC_XCONCAT(__clc_vstorea_half, __CLC_VECSIZE), x)

#ifndef __CLC_SCALAR

#define __CLC_VSTORE_DECL(ADDRSPACE)                                           \
  _CLC_OVERLOAD _CLC_DECL void __CLC_VSTORE_NAME(                              \
      __CLC_VSTORE_TY data, size_t offset, ADDRSPACE __CLC_SCALAR_GENTYPE *p);

__CLC_VSTORE_DECL(__private)
__CLC_VSTORE_DECL(__local)
__CLC_VSTORE_DECL(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VSTORE_DECL(__generic)
#endif

#undef __CLC_VSTORE_DECL

#endif // __CLC_SCALAR

// vstore_half and vstorea_half are available even if cl_khr_fp16 is
// unavailable.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32 || __CLC_FPSIZE == 64

#define __CLC_VSTORE_HALF_DECL(ADDRSPACE, SUFFIX)                              \
  _CLC_OVERLOAD _CLC_DECL void __CLC_VSTORE_HALF_NAME(SUFFIX)(                 \
      __CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p);                 \
                                                                               \
  _CLC_OVERLOAD _CLC_DECL void __CLC_VSTOREA_HALF_NAME(SUFFIX)(                \
      __CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p);

#define __CLC_VSTORE_HALF_DECL_ALL_MODES(ADDRSPACE)                            \
  __CLC_VSTORE_HALF_DECL(ADDRSPACE, )                                          \
  __CLC_VSTORE_HALF_DECL(ADDRSPACE, _rtz)                                      \
  __CLC_VSTORE_HALF_DECL(ADDRSPACE, _rtn)                                      \
  __CLC_VSTORE_HALF_DECL(ADDRSPACE, _rtp)                                      \
  __CLC_VSTORE_HALF_DECL(ADDRSPACE, _rte)

__CLC_VSTORE_HALF_DECL_ALL_MODES(__private)
__CLC_VSTORE_HALF_DECL_ALL_MODES(__local)
__CLC_VSTORE_HALF_DECL_ALL_MODES(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VSTORE_HALF_DECL_ALL_MODES(__generic)
#endif

#undef __CLC_VSTORE_HALF_DECL
#undef __CLC_VSTORE_HALF_DECL_ALL_MODES

#endif
#endif

#undef __CLC_VSTORE_TY
#undef __CLC_VSTORE_NAME
#undef __CLC_VSTORE_HALF_NAME
#undef __CLC_VSTOREA_HALF_NAME
