//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __CLC_VSTORE_TY __CLC_XCONCAT(less_aligned_, __CLC_GENTYPE)
#define __CLC_VSTORE_NAME(x)                                                   \
  __CLC_XCONCAT(__CLC_XCONCAT(x, vstore), __CLC_VECSIZE)
#define __CLC_VSTORE_HALF_NAME(x, y)                                           \
  __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(x, vstore_half), __CLC_VECSIZE), y)
#define __CLC_VSTOREA_HALF_NAME(x, y)                                          \
  __CLC_XCONCAT(__CLC_XCONCAT(__CLC_XCONCAT(x, vstorea_half), __CLC_VECSIZE), y)

#ifndef __CLC_SCALAR

#define __CLC_VSTORE_DEF(ADDRSPACE)                                            \
  _CLC_OVERLOAD _CLC_DEF void __CLC_VSTORE_NAME()(                             \
      __CLC_VSTORE_TY data, size_t offset,                                     \
      ADDRSPACE __CLC_SCALAR_GENTYPE *p) {                                     \
    return __CLC_VSTORE_NAME(__clc_)(data, offset, p);                         \
  }

__CLC_VSTORE_DEF(__private)
__CLC_VSTORE_DEF(__local)
__CLC_VSTORE_DEF(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VSTORE_DEF(__generic)
#endif

#undef __CLC_VSTORE_DEF

#endif // __CLC_SCALAR

// vstore_half and vstorea_half are available even if cl_khr_fp16 is
// unavailable.
#ifdef __CLC_FPSIZE
#if __CLC_FPSIZE == 32 || __CLC_FPSIZE == 64

#define __CLC_VSTORE_HALF_DEF(ADDRSPACE, SUFFIX)                               \
  _CLC_OVERLOAD _CLC_DEF void __CLC_VSTORE_HALF_NAME(, SUFFIX)(                \
      __CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p) {                \
    __CLC_VSTORE_HALF_NAME(__clc_, SUFFIX)(data, offset, p);                   \
  }                                                                            \
                                                                               \
  _CLC_OVERLOAD _CLC_DEF void __CLC_VSTOREA_HALF_NAME(, SUFFIX)(               \
      __CLC_VSTORE_TY data, size_t offset, ADDRSPACE half *p) {                \
    __CLC_VSTOREA_HALF_NAME(__clc_, SUFFIX)(data, offset, p);                  \
  }

#define __CLC_VSTORE_HALF_DEF_ALL_MODES(ADDRSPACE)                             \
  __CLC_VSTORE_HALF_DEF(ADDRSPACE, )                                           \
  __CLC_VSTORE_HALF_DEF(ADDRSPACE, _rtz)                                       \
  __CLC_VSTORE_HALF_DEF(ADDRSPACE, _rtn)                                       \
  __CLC_VSTORE_HALF_DEF(ADDRSPACE, _rtp)                                       \
  __CLC_VSTORE_HALF_DEF(ADDRSPACE, _rte)

__CLC_VSTORE_HALF_DEF_ALL_MODES(__private)
__CLC_VSTORE_HALF_DEF_ALL_MODES(__local)
__CLC_VSTORE_HALF_DEF_ALL_MODES(__global)

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
__CLC_VSTORE_HALF_DEF_ALL_MODES(__generic)
#endif

#undef __CLC_VSTORE_HALF_DEF
#undef __CLC_VSTORE_HALF_DEF_ALL_MODES

#endif
#endif

#undef __CLC_VSTORE_TY
#undef __CLC_VSTORE_NAME
#undef __CLC_VSTORE_HALF_NAME
#undef __CLC_VSTOREA_HALF_NAME
