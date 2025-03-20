#include <clc/clcmacro.h>
#include <clc/integer/clc_clz.h>
#include <clc/internal/clc.h>

_CLC_OVERLOAD _CLC_DEF char __clc_clz(char x) {
  return __clc_clz((ushort)(uchar)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF uchar __clc_clz(uchar x) {
  return __clc_clz((ushort)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF short __clc_clz(short x) {
  return x ? __builtin_clzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF ushort __clc_clz(ushort x) {
  return x ? __builtin_clzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF int __clc_clz(int x) {
  return x ? __builtin_clz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF uint __clc_clz(uint x) {
  return x ? __builtin_clz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF long __clc_clz(long x) {
  return x ? __builtin_clzl(x) : 64;
}

_CLC_OVERLOAD _CLC_DEF ulong __clc_clz(ulong x) {
  return x ? __builtin_clzl(x) : 64;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, __clc_clz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, __clc_clz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, __clc_clz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, __clc_clz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, __clc_clz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, __clc_clz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, __clc_clz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, __clc_clz, ulong)
