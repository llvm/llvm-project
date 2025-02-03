#include <clc/integer/clc_hadd.h>
#include <clc/integer/definitions.h>
#include <clc/internal/clc.h>

// TODO: Replace with __clc_convert_<type> when available
#define __CLC_CONVERT_TY(X, TY) __builtin_convertvector(X, TY)

#define __CLC_MUL_HI_VEC_IMPL(BGENTYPE, GENTYPE, GENSIZE)                      \
  _CLC_OVERLOAD _CLC_DEF GENTYPE __clc_mul_hi(GENTYPE x, GENTYPE y) {          \
    BGENTYPE large_x = __CLC_CONVERT_TY(x, BGENTYPE);                          \
    BGENTYPE large_y = __CLC_CONVERT_TY(y, BGENTYPE);                          \
    BGENTYPE large_mul_hi = (large_x * large_y) >> (BGENTYPE)GENSIZE;          \
    return __CLC_CONVERT_TY(large_mul_hi, GENTYPE);                            \
  }

// For all types EXCEPT long, which is implemented separately
#define __CLC_MUL_HI_IMPL(BGENTYPE, GENTYPE, GENSIZE)                          \
  _CLC_OVERLOAD _CLC_DEF GENTYPE __clc_mul_hi(GENTYPE x, GENTYPE y) {          \
    return (GENTYPE)(((BGENTYPE)x * (BGENTYPE)y) >> GENSIZE);                  \
  }

#define __CLC_MUL_HI_DEC_IMPL(BTYPE, TYPE, BITS)                               \
  __CLC_MUL_HI_IMPL(BTYPE, TYPE, BITS)                                         \
  __CLC_MUL_HI_VEC_IMPL(BTYPE##2, TYPE##2, BITS)                               \
  __CLC_MUL_HI_VEC_IMPL(BTYPE##3, TYPE##3, BITS)                               \
  __CLC_MUL_HI_VEC_IMPL(BTYPE##4, TYPE##4, BITS)                               \
  __CLC_MUL_HI_VEC_IMPL(BTYPE##8, TYPE##8, BITS)                               \
  __CLC_MUL_HI_VEC_IMPL(BTYPE##16, TYPE##16, BITS)

_CLC_OVERLOAD _CLC_DEF long __clc_mul_hi(long x, long y) {
  long f, o, i;
  ulong l;

  // Move the high/low halves of x/y into the lower 32-bits of variables so
  // that we can multiply them without worrying about overflow.
  long x_hi = x >> 32;
  long x_lo = x & UINT_MAX;
  long y_hi = y >> 32;
  long y_lo = y & UINT_MAX;

  // Multiply all of the components according to FOIL method
  f = x_hi * y_hi;
  o = x_hi * y_lo;
  i = x_lo * y_hi;
  l = x_lo * y_lo;

  // Now add the components back together in the following steps:
  // F: doesn't need to be modified
  // O/I: Need to be added together.
  // L: Shift right by 32-bits, then add into the sum of O and I
  // Once O/I/L are summed up, then shift the sum by 32-bits and add to F.
  //
  // We use hadd to give us a bit of extra precision for the intermediate sums
  // but as a result, we shift by 31 bits instead of 32
  return (long)(f + (__clc_hadd(o, (i + (long)((ulong)l >> 32))) >> 31));
}

_CLC_OVERLOAD _CLC_DEF ulong __clc_mul_hi(ulong x, ulong y) {
  ulong f, o, i;
  ulong l;

  // Move the high/low halves of x/y into the lower 32-bits of variables so
  // that we can multiply them without worrying about overflow.
  ulong x_hi = x >> 32;
  ulong x_lo = x & UINT_MAX;
  ulong y_hi = y >> 32;
  ulong y_lo = y & UINT_MAX;

  // Multiply all of the components according to FOIL method
  f = x_hi * y_hi;
  o = x_hi * y_lo;
  i = x_lo * y_hi;
  l = x_lo * y_lo;

  // Now add the components back together, taking care to respect the fact that:
  // F: doesn't need to be modified
  // O/I: Need to be added together.
  // L: Shift right by 32-bits, then add into the sum of O and I
  // Once O/I/L are summed up, then shift the sum by 32-bits and add to F.
  //
  // We use hadd to give us a bit of extra precision for the intermediate sums
  // but as a result, we shift by 31 bits instead of 32
  return (f + (__clc_hadd(o, (i + (l >> 32))) >> 31));
}

// Vector-based mul_hi implementation for logn/ulong. See comments in the scalar
// versions for more detail.
#define __CLC_MUL_HI_LONG_VEC_IMPL(TY, UTY)                                    \
  _CLC_OVERLOAD _CLC_DEF TY __clc_mul_hi(TY x, TY y) {                         \
    TY f, o, i;                                                                \
    UTY l;                                                                     \
                                                                               \
    TY x_hi = x >> 32;                                                         \
    TY x_lo = x & UINT_MAX;                                                    \
    TY y_hi = y >> 32;                                                         \
    TY y_lo = y & UINT_MAX;                                                    \
                                                                               \
    f = x_hi * y_hi;                                                           \
    o = x_hi * y_lo;                                                           \
    i = x_lo * y_hi;                                                           \
    l = __CLC_CONVERT_TY(x_lo * y_lo, UTY);                                    \
    i += __CLC_CONVERT_TY(l >> (UTY)32, TY);                                   \
                                                                               \
    return f + (__clc_hadd(o, i) >> (TY)31);                                   \
  }

#define __CLC_MUL_HI_LONG_IMPL(BTYPE, UBTYPE)                                  \
  __CLC_MUL_HI_LONG_VEC_IMPL(BTYPE##2, UBTYPE##2)                              \
  __CLC_MUL_HI_LONG_VEC_IMPL(BTYPE##3, UBTYPE##3)                              \
  __CLC_MUL_HI_LONG_VEC_IMPL(BTYPE##4, UBTYPE##4)                              \
  __CLC_MUL_HI_LONG_VEC_IMPL(BTYPE##8, UBTYPE##8)                              \
  __CLC_MUL_HI_LONG_VEC_IMPL(BTYPE##16, UBTYPE##16)

#define __CLC_MUL_HI_TYPES()                                                   \
  __CLC_MUL_HI_DEC_IMPL(short, char, 8)                                        \
  __CLC_MUL_HI_DEC_IMPL(ushort, uchar, 8)                                      \
  __CLC_MUL_HI_DEC_IMPL(int, short, 16)                                        \
  __CLC_MUL_HI_DEC_IMPL(uint, ushort, 16)                                      \
  __CLC_MUL_HI_DEC_IMPL(long, int, 32)                                         \
  __CLC_MUL_HI_DEC_IMPL(ulong, uint, 32)                                       \
  __CLC_MUL_HI_LONG_IMPL(long, ulong)                                          \
  __CLC_MUL_HI_LONG_IMPL(ulong, ulong)

__CLC_MUL_HI_TYPES()

#undef __CLC_MUL_HI_TYPES
#undef __CLC_MUL_HI_LONG_IMPL
#undef __CLC_MUL_HI_LONG_VEC_IMPL
#undef __CLC_MUL_HI_DEC_IMPL
#undef __CLC_MUL_HI_IMPL
#undef __CLC_MUL_HI_VEC_IMPL
#undef __CLC_CONVERT_TY
