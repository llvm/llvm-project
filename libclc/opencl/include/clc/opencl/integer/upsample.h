//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_OPENCL_INTEGER_UPSAMPLE_H__
#define __CLC_OPENCL_INTEGER_UPSAMPLE_H__

#include <clc/opencl/opencl-base.h>

#define __CLC_UPSAMPLE_DECL(BGENTYPE, GENTYPE, UGENTYPE)                       \
  _CLC_OVERLOAD _CLC_CONST _CLC_DECL BGENTYPE upsample(GENTYPE hi, UGENTYPE lo);

#define __CLC_UPSAMPLE_VEC(BGENTYPE, GENTYPE, UGENTYPE)                        \
  __CLC_UPSAMPLE_DECL(BGENTYPE, GENTYPE, UGENTYPE)                             \
  __CLC_UPSAMPLE_DECL(BGENTYPE##2, GENTYPE##2, UGENTYPE##2)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##3, GENTYPE##3, UGENTYPE##3)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##4, GENTYPE##4, UGENTYPE##4)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##8, GENTYPE##8, UGENTYPE##8)                    \
  __CLC_UPSAMPLE_DECL(BGENTYPE##16, GENTYPE##16, UGENTYPE##16)

#define __CLC_UPSAMPLE_TYPES()                                                 \
  __CLC_UPSAMPLE_VEC(short, char, uchar)                                       \
  __CLC_UPSAMPLE_VEC(ushort, uchar, uchar)                                     \
  __CLC_UPSAMPLE_VEC(int, short, ushort)                                       \
  __CLC_UPSAMPLE_VEC(uint, ushort, ushort)                                     \
  __CLC_UPSAMPLE_VEC(long, int, uint)                                          \
  __CLC_UPSAMPLE_VEC(ulong, uint, uint)

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_DECL
#undef __CLC_UPSAMPLE_VEC

#endif // __CLC_OPENCL_INTEGER_UPSAMPLE_H__
