//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/integer/clc_clz.h>
#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_DEF char __clc_clz(char x) {
  return __clc_clz(__clc_as_uchar(x));
}

_CLC_OVERLOAD _CLC_DEF uchar __clc_clz(uchar x) {
  return __builtin_clzg(x, 8);
}

_CLC_OVERLOAD _CLC_DEF short __clc_clz(short x) {
  return __clc_clz(__clc_as_ushort(x));
}

_CLC_OVERLOAD _CLC_DEF ushort __clc_clz(ushort x) {
  return __builtin_clzg(x, 16);
}

_CLC_OVERLOAD _CLC_DEF int __clc_clz(int x) {
  return __clc_clz(__clc_as_uint(x));
}

_CLC_OVERLOAD _CLC_DEF uint __clc_clz(uint x) {
  return __builtin_clzg(x, 32);
}

_CLC_OVERLOAD _CLC_DEF long __clc_clz(long x) {
  return __clc_clz(__clc_as_ulong(x));
}

_CLC_OVERLOAD _CLC_DEF ulong __clc_clz(ulong x) {
  return __builtin_clzg(x, 64);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __clc_clz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __clc_clz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __clc_clz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_clz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __clc_clz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_clz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __clc_clz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __clc_clz, ulong)
