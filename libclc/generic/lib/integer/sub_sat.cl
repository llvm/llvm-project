#include <clc/clc.h>
#include "../clcmacro.h"

// From sub_sat.ll
_CLC_DECL char   __clc_sub_sat_s8(char, char);
_CLC_DECL uchar  __clc_sub_sat_u8(uchar, uchar);
_CLC_DECL short  __clc_sub_sat_s16(short, short);
_CLC_DECL ushort __clc_sub_sat_u16(ushort, ushort);
_CLC_DECL int    __clc_sub_sat_s32(int, int);
_CLC_DECL uint   __clc_sub_sat_u32(uint, uint);
_CLC_DECL long   __clc_sub_sat_s64(long, long);
_CLC_DECL ulong  __clc_sub_sat_u64(ulong, ulong);

_CLC_OVERLOAD _CLC_DEF char sub_sat(char x, char y) {
  return __clc_sub_sat_s8(x, y);
}

_CLC_OVERLOAD _CLC_DEF uchar sub_sat(uchar x, uchar y) {
  return __clc_sub_sat_u8(x, y);
}

_CLC_OVERLOAD _CLC_DEF short sub_sat(short x, short y) {
  return __clc_sub_sat_s16(x, y);
}

_CLC_OVERLOAD _CLC_DEF ushort sub_sat(ushort x, ushort y) {
  return __clc_sub_sat_u16(x, y);
}

_CLC_OVERLOAD _CLC_DEF int sub_sat(int x, int y) {
  return __clc_sub_sat_s32(x, y);
}

_CLC_OVERLOAD _CLC_DEF uint sub_sat(uint x, uint y) {
  return __clc_sub_sat_u32(x, y);
}

_CLC_OVERLOAD _CLC_DEF long sub_sat(long x, long y) {
  return __clc_sub_sat_s64(x, y);
}

_CLC_OVERLOAD _CLC_DEF ulong sub_sat(ulong x, ulong y) {
  return __clc_sub_sat_u64(x, y);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, sub_sat, char, char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, sub_sat, uchar, uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, sub_sat, short, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, sub_sat, ushort, ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, sub_sat, int, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, sub_sat, uint, uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, sub_sat, long, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, sub_sat, ulong, ulong)
