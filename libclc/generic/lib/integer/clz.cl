#include <clc/clc.h>
#include "../clcmacro.h"

// From clz.ll
_CLC_DECL char   __clc_clz_s8(char);
_CLC_DECL uchar  __clc_clz_u8(uchar);
_CLC_DECL short  __clc_clz_s16(short);
_CLC_DECL ushort __clc_clz_u16(ushort);
_CLC_DECL int    __clc_clz_s32(int);
_CLC_DECL uint   __clc_clz_u32(uint);
_CLC_DECL long   __clc_clz_s64(long);
_CLC_DECL ulong  __clc_clz_u64(ulong);

_CLC_OVERLOAD _CLC_DEF char clz(char x) {
  return __clc_clz_s8(x);
}

_CLC_OVERLOAD _CLC_DEF uchar clz(uchar x) {
  return __clc_clz_u8(x);
}

_CLC_OVERLOAD _CLC_DEF short clz(short x) {
  return __clc_clz_s16(x);
}

_CLC_OVERLOAD _CLC_DEF ushort clz(ushort x) {
  return __clc_clz_u16(x);
}

_CLC_OVERLOAD _CLC_DEF int clz(int x) {
  return __clc_clz_s32(x);
}

_CLC_OVERLOAD _CLC_DEF uint clz(uint x) {
  return __clc_clz_u32(x);
}

_CLC_OVERLOAD _CLC_DEF long clz(long x) {
  return __clc_clz_s64(x);
}

_CLC_OVERLOAD _CLC_DEF ulong clz(ulong x) {
  return __clc_clz_u64(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, clz, char)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, clz, uchar)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, clz, short)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, clz, ushort)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, clz, int)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, clz, uint)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, clz, long)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, clz, ulong)
