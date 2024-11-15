#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF char clz(char x) {
  return clz((ushort)(uchar)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF uchar clz(uchar x) {
  return clz((ushort)x) - 8;
}

_CLC_OVERLOAD _CLC_DEF short clz(short x) {
  return x ? __builtin_clzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF ushort clz(ushort x) {
  return x ? __builtin_clzs(x) : 16;
}

_CLC_OVERLOAD _CLC_DEF int clz(int x) {
  return x ? __builtin_clz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF uint clz(uint x) {
  return x ? __builtin_clz(x) : 32;
}

_CLC_OVERLOAD _CLC_DEF long clz(long x) {
  return x ? __builtin_clzl(x) : 64;
}

_CLC_OVERLOAD _CLC_DEF ulong clz(ulong x) {
  return x ? __builtin_clzl(x) : 64;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, clz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, clz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, clz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, clz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, clz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, clz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, clz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, clz, ulong)
