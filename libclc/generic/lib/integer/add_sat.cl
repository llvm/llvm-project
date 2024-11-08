#include <clc/clc.h>
#include <clc/clcmacro.h>

// From add_sat.ll
_CLC_DECL char   __clc_add_sat_s8(char, char);
_CLC_DECL uchar  __clc_add_sat_u8(uchar, uchar);
_CLC_DECL short  __clc_add_sat_s16(short, short);
_CLC_DECL ushort __clc_add_sat_u16(ushort, ushort);
_CLC_DECL int    __clc_add_sat_s32(int, int);
_CLC_DECL uint   __clc_add_sat_u32(uint, uint);
_CLC_DECL long   __clc_add_sat_s64(long, long);
_CLC_DECL ulong  __clc_add_sat_u64(ulong, ulong);

_CLC_OVERLOAD _CLC_DEF char add_sat(char x, char y) {
  short r = x + y;
  return convert_char_sat(r);
}

_CLC_OVERLOAD _CLC_DEF uchar add_sat(uchar x, uchar y) {
  ushort r = x + y;
  return convert_uchar_sat(r);
}

_CLC_OVERLOAD _CLC_DEF short add_sat(short x, short y) {
  int r = x + y;
  return convert_short_sat(r);
}

_CLC_OVERLOAD _CLC_DEF ushort add_sat(ushort x, ushort y) {
  uint r = x + y;
  return convert_ushort_sat(r);
}

_CLC_OVERLOAD _CLC_DEF int add_sat(int x, int y) {
  int r;
  if (__builtin_sadd_overflow(x, y, &r))
    // The oveflow can only occur if both are pos or both are neg,
    // thus we only need to check one operand
    return x > 0 ? INT_MAX : INT_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF uint add_sat(uint x, uint y) {
  uint r;
  if (__builtin_uadd_overflow(x, y, &r))
	return UINT_MAX;
  return r;
}

_CLC_OVERLOAD _CLC_DEF long add_sat(long x, long y) {
  long r;
  if (__builtin_saddl_overflow(x, y, &r))
    // The oveflow can only occur if both are pos or both are neg,
    // thus we only need to check one operand
    return x > 0 ? LONG_MAX : LONG_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF ulong add_sat(ulong x, ulong y) {
  ulong r;
  if (__builtin_uaddl_overflow(x, y, &r))
	return ULONG_MAX;
  return r;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, add_sat, char, char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, add_sat, uchar, uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, add_sat, short, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, add_sat, ushort, ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, add_sat, int, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, add_sat, uint, uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, add_sat, long, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, add_sat, ulong, ulong)
