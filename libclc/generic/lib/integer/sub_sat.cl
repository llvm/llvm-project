#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF char sub_sat(char x, char y) {
  short r = x - y;
  return convert_char_sat(r);
}

_CLC_OVERLOAD _CLC_DEF uchar sub_sat(uchar x, uchar y) {
  short r = x - y;
  return convert_uchar_sat(r);
}

_CLC_OVERLOAD _CLC_DEF short sub_sat(short x, short y) {
  int r = x - y;
  return convert_short_sat(r);
}

_CLC_OVERLOAD _CLC_DEF ushort sub_sat(ushort x, ushort y) {
  int r = x - y;
  return convert_ushort_sat(r);
}

_CLC_OVERLOAD _CLC_DEF int sub_sat(int x, int y) {
  int r;
  if (__builtin_ssub_overflow(x, y, &r))
    // The oveflow can only occur in the direction of the first operand
    return x > 0 ? INT_MAX : INT_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF uint sub_sat(uint x, uint y) {
  uint r;
  if (__builtin_usub_overflow(x, y, &r))
	return 0;
  return r;
}

_CLC_OVERLOAD _CLC_DEF long sub_sat(long x, long y) {
  long r;
  if (__builtin_ssubl_overflow(x, y, &r))
    // The oveflow can only occur in the direction of the first operand
    return x > 0 ? LONG_MAX : LONG_MIN;
  return r;
}

_CLC_OVERLOAD _CLC_DEF ulong sub_sat(ulong x, ulong y) {
  ulong r;
  if (__builtin_usubl_overflow(x, y, &r))
	return 0;
  return r;
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, sub_sat, char, char)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, sub_sat, uchar, uchar)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, sub_sat, short, short)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, sub_sat, ushort, ushort)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, sub_sat, int, int)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, sub_sat, uint, uint)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, sub_sat, long, long)
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, sub_sat, ulong, ulong)
