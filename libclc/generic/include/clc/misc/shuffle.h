//===-- generic/include/clc/misc/shuffle.h ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under both the University of Illinois Open Source
// License and the MIT license. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define _CLC_SHUFFLE_DECL(TYPE, MASKTYPE, RETTYPE) \
  _CLC_OVERLOAD _CLC_DECL RETTYPE shuffle(TYPE x, MASKTYPE mask);

//Return type is same base type as the input type, with the same vector size as the mask.
//Elements in the mask must be the same size (number of bits) as the input value.
//E.g. char8 ret = shuffle(char2 x, uchar8 mask);

#define _CLC_VECTOR_SHUFFLE_MASKSIZE(INBASE, INTYPE, MASKTYPE) \
  _CLC_SHUFFLE_DECL(INTYPE, MASKTYPE##2, INBASE##2) \
  _CLC_SHUFFLE_DECL(INTYPE, MASKTYPE##4, INBASE##4) \
  _CLC_SHUFFLE_DECL(INTYPE, MASKTYPE##8, INBASE##8) \
  _CLC_SHUFFLE_DECL(INTYPE, MASKTYPE##16, INBASE##16) \

#define _CLC_VECTOR_SHUFFLE_INSIZE(TYPE, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##2, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##4, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##8, MASKTYPE) \
  _CLC_VECTOR_SHUFFLE_MASKSIZE(TYPE, TYPE##16, MASKTYPE) \

_CLC_VECTOR_SHUFFLE_INSIZE(char, uchar)
_CLC_VECTOR_SHUFFLE_INSIZE(short, ushort)
_CLC_VECTOR_SHUFFLE_INSIZE(int, uint)
_CLC_VECTOR_SHUFFLE_INSIZE(long, ulong)
_CLC_VECTOR_SHUFFLE_INSIZE(uchar, uchar)
_CLC_VECTOR_SHUFFLE_INSIZE(ushort, ushort)
_CLC_VECTOR_SHUFFLE_INSIZE(uint, uint)
_CLC_VECTOR_SHUFFLE_INSIZE(ulong, ulong)
_CLC_VECTOR_SHUFFLE_INSIZE(float, uint)
#ifdef cl_khr_fp64
_CLC_VECTOR_SHUFFLE_INSIZE(double, ulong)
#endif
#ifdef cl_khr_fp16
_CLC_VECTOR_SHUFFLE_INSIZE(half, ushort)
#endif

#undef _CLC_SHUFFLE_DECL
#undef _CLC_VECTOR_SHUFFLE_MASKSIZE
#undef _CLC_VECTOR_SHUFFLE_INSIZE
