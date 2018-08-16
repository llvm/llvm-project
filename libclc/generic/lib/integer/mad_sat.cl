#include <clc/clc.h>
#include "../clcmacro.h"

_CLC_OVERLOAD _CLC_DEF char mad_sat(char x, char y, char z) {
  return clamp((short)mad24((short)x, (short)y, (short)z), (short)CHAR_MIN, (short) CHAR_MAX);
}

_CLC_OVERLOAD _CLC_DEF uchar mad_sat(uchar x, uchar y, uchar z) {
  return clamp((ushort)mad24((ushort)x, (ushort)y, (ushort)z), (ushort)0, (ushort) UCHAR_MAX);
}

_CLC_OVERLOAD _CLC_DEF short mad_sat(short x, short y, short z) {
  return clamp((int)mad24((int)x, (int)y, (int)z), (int)SHRT_MIN, (int) SHRT_MAX);
}

_CLC_OVERLOAD _CLC_DEF ushort mad_sat(ushort x, ushort y, ushort z) {
  return clamp((uint)mad24((uint)x, (uint)y, (uint)z), (uint)0, (uint) USHRT_MAX);
}

_CLC_OVERLOAD _CLC_DEF int mad_sat(int x, int y, int z) {
  int mhi = mul_hi(x, y);
  uint mlo = x * y;
  long m = upsample(mhi, mlo);
  m += z;
  if (m > INT_MAX)
    return INT_MAX;
  if (m < INT_MIN)
    return INT_MIN;
  return m;
}

_CLC_OVERLOAD _CLC_DEF uint mad_sat(uint x, uint y, uint z) {
  if (mul_hi(x, y) != 0)
    return UINT_MAX;
  return add_sat(x * y, z);
}

_CLC_OVERLOAD _CLC_DEF long mad_sat(long x, long y, long z) {
  long hi = mul_hi(x, y);
  ulong ulo = x * y;
  long  slo = x * y;
  /* Big overflow of more than 2 bits, add can't fix this */
  if (((x < 0) == (y < 0)) && hi != 0)
    return LONG_MAX;
  /* Low overflow in mul and z not neg enough to correct it */
  if (hi == 0 && ulo >= LONG_MAX && (z > 0 || (ulo + z) > LONG_MAX))
    return LONG_MAX;
  /* Big overflow of more than 2 bits, add can't fix this */
  if (((x < 0) != (y < 0)) && hi != -1)
    return LONG_MIN;
  /* Low overflow in mul and z not pos enough to correct it */
  if (hi == -1 && ulo <= ((ulong)LONG_MAX + 1UL) && (z < 0 || z < (LONG_MAX - ulo)))
    return LONG_MIN;
  /* We have checked all conditions, any overflow in addition returns
   * the correct value */
  return ulo + z;
}

_CLC_OVERLOAD _CLC_DEF ulong mad_sat(ulong x, ulong y, ulong z) {
  if (mul_hi(x, y) != 0)
    return ULONG_MAX;
  return add_sat(x * y, z);
}

_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, char, mad_sat, char, char, char)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uchar, mad_sat, uchar, uchar, uchar)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, short, mad_sat, short, short, short)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ushort, mad_sat, ushort, ushort, ushort)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, int, mad_sat, int, int, int)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, uint, mad_sat, uint, uint, uint)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, mad_sat, long, long, long)
_CLC_TERNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, ulong, mad_sat, ulong, ulong, ulong)
