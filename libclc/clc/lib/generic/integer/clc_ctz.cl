//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clcmacro.h>
#include <clc/integer/clc_ctz.h>
#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_DEF char __clc_ctz(char x) {
  return __clc_ctz(__clc_as_uchar(x));
}

_CLC_OVERLOAD _CLC_DEF uchar __clc_ctz(uchar x) { return __builtin_ctzg(x, 8); }

_CLC_OVERLOAD _CLC_DEF short __clc_ctz(short x) {
  return __clc_ctz(__clc_as_ushort(x));
}

_CLC_OVERLOAD _CLC_DEF ushort __clc_ctz(ushort x) {
  return __builtin_ctzg(x, 16);
}

_CLC_OVERLOAD _CLC_DEF int __clc_ctz(int x) {
  return __clc_ctz(__clc_as_uint(x));
}

_CLC_OVERLOAD _CLC_DEF uint __clc_ctz(uint x) { return __builtin_ctzg(x, 32); }

_CLC_OVERLOAD _CLC_DEF long __clc_ctz(long x) {
  return __clc_ctz(__clc_as_ulong(x));
}

_CLC_OVERLOAD _CLC_DEF ulong __clc_ctz(ulong x) {
  return __builtin_ctzg(x, 64);
}

#define __CLC_FUNCTION __clc_ctz
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/integer/gentype.inc>
