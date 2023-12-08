//===-- generic/include/clc/misc/shuffle2.h ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define _CLC_SHUFFLE2_DECL(TYPE, MASKTYPE, RETTYPE) \
  _CLC_OVERLOAD _CLC_DECL RETTYPE shuffle2(TYPE x, TYPE y, MASKTYPE mask);

//Return type is same base type as the input type, with the same vector size as the mask.
//Elements in the mask must be the same size (number of bits) as the input value.
//E.g. char8 ret = shuffle2(char2 x, char2 y, uchar8 mask);

#define _CLC_VECTOR_SHUFFLE2_MASKSIZE(INBASE, INTYPE, MASKTYPE) \
  _CLC_SHUFFLE2_DECL(INTYPE, MASKTYPE##2, INBASE##2) \
  _CLC_SHUFFLE2_DECL(INTYPE, MASKTYPE##4, INBASE##4) \
  _CLC_SHUFFLE2_DECL(INTYPE, MASKTYPE##8, INBASE##8) \
  _CLC_SHUFFLE2_DECL(INTYPE, MASKTYPE##16, INBASE##16) \

#define _CLC_VECTOR_SHUFFLE2_INSIZE(TYPE, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE2_MASKSIZE(TYPE, TYPE##2, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE2_MASKSIZE(TYPE, TYPE##4, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE2_MASKSIZE(TYPE, TYPE##8, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE2_MASKSIZE(TYPE, TYPE##16, MASKTYPE) \

_CLC_VECTOR_SHUFFLE2_INSIZE(char, uchar)
_CLC_VECTOR_SHUFFLE2_INSIZE(short, ushort)
_CLC_VECTOR_SHUFFLE2_INSIZE(int, uint)
_CLC_VECTOR_SHUFFLE2_INSIZE(long, ulong)
_CLC_VECTOR_SHUFFLE2_INSIZE(uchar, uchar)
_CLC_VECTOR_SHUFFLE2_INSIZE(ushort, ushort)
_CLC_VECTOR_SHUFFLE2_INSIZE(uint, uint)
_CLC_VECTOR_SHUFFLE2_INSIZE(ulong, ulong)
_CLC_VECTOR_SHUFFLE2_INSIZE(float, uint)
#ifdef cl_khr_fp64
_CLC_VECTOR_SHUFFLE2_INSIZE(double, ulong)
#endif
#ifdef cl_khr_fp16
_CLC_VECTOR_SHUFFLE2_INSIZE(half, ushort)
#endif

#undef _CLC_SHUFFLE_DECL
#undef _CLC_VECTOR_SHUFFLE2_MASKSIZE
#undef _CLC_VECTOR_SHUFFLE2_INSIZE
