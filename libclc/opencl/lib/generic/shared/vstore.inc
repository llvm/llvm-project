//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define CLC_VSTORE_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)
#define CLC_VSTORE_NAME(x)                                                     \
  __CLC_XCONCAT(__CLC_XCONCAT(x, vstore), __CLC_VECSIZE)
#define CLC_VSTORE_HALF_NAME(x, y)                                             \
  __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(x, vstore_half), __CLC_VECSIZE), y)
#define CLC_VSTOREA_HALF_NAME(x, y)                                            \
  __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(x, vstorea_half), __CLC_VECSIZE), y)

#ifndef __CLC_SCALAR

#define CLC_VSTORE_DEF(ADDRSPACE)                                              \
  _CLC_OVERLOAD _CLC_DEF void CLC_VSTORE_NAME()(                               \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE __CLC_SCALAR_GENTYPE *p) {  \
    return CLC_VSTORE_NAME(__clc_)(data, offset, p);                           \
  }

CLC_VSTORE_DEF(__private)
CLC_VSTORE_DEF(__local)
CLC_VSTORE_DEF(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
CLC_VSTORE_DEF(__generic)
#endif

#undef CLC_VSTORE_DEF

#endif // __CLC_SCALAR

// vstore_half and vstorea_half are available even if cl_khr_fp16 is
// unavailable.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32 || __CLC_FPSIZE == 64

#define CLC_VSTORE_HALF_DEF(ADDRSPACE, SUFFIX)                                 \
  _CLC_OVERLOAD _CLC_DEF void CLC_VSTORE_HALF_NAME(, SUFFIX)(                  \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p) {                  \
    CLC_VSTORE_HALF_NAME(__clc_, SUFFIX)(data, offset, p);                     \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF void CLC_VSTOREA_HALF_NAME(, SUFFIX)(                 \
      CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p) {                  \
    CLC_VSTOREA_HALF_NAME(__clc_, SUFFIX)(data, offset, p);                    \
  }

#define CLC_VSTORE_HALF_DEF_ALL_MODES(ADDRSPACE)                               \
  CLC_VSTORE_HALF_DEF(ADDRSPACE, )                                             \
  CLC_VSTORE_HALF_DEF(ADDRSPACE, _rtz)                                         \
  CLC_VSTORE_HALF_DEF(ADDRSPACE, _rtn)                                         \
  CLC_VSTORE_HALF_DEF(ADDRSPACE, _rtp)                                         \
  CLC_VSTORE_HALF_DEF(ADDRSPACE, _rte)

CLC_VSTORE_HALF_DEF_ALL_MODES(__private)
CLC_VSTORE_HALF_DEF_ALL_MODES(__local)
CLC_VSTORE_HALF_DEF_ALL_MODES(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
CLC_VSTORE_HALF_DEF_ALL_MODES(__generic)
#endif

#undef CLC_VSTORE_HALF_DEF
#undef CLC_VSTORE_HALF_DEF_ALL_MODES

#endif
#endif

#undef CLC_VSTORE_TY
#undef CLC_VSTORE_NAME
#undef CLC_VSTORE_HALF_NAME
#undef CLC_VSTOREA_HALF_NAME
