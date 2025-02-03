#include <clc/internal/clc.h>

// TODO: Replace with __clc_convert_<type> when available
#define __CLC_CONVERT_TY(X, TY) __builtin_convertvector(X, TY)

#define __CLC_UPSAMPLE_VEC_IMPL(BGENTYPE, GENTYPE, UGENTYPE, GENSIZE)          \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE __clc_upsample(GENTYPE hi, UGENTYPE lo) {    \
    BGENTYPE large_hi = __CLC_CONVERT_TY(hi, BGENTYPE);                        \
    BGENTYPE large_lo = __CLC_CONVERT_TY(lo, BGENTYPE);                        \
    return (large_hi << (BGENTYPE)GENSIZE) | large_lo;                         \
  }

#define __CLC_UPSAMPLE_IMPL(BGENTYPE, GENTYPE, UGENTYPE, GENSIZE)              \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE __clc_upsample(GENTYPE hi, UGENTYPE lo) {    \
    return ((BGENTYPE)hi << GENSIZE) | lo;                                     \
  }                                                                            \
  __CLC_UPSAMPLE_VEC_IMPL(BGENTYPE##2, GENTYPE##2, UGENTYPE##2, GENSIZE)       \
  __CLC_UPSAMPLE_VEC_IMPL(BGENTYPE##3, GENTYPE##3, UGENTYPE##3, GENSIZE)       \
  __CLC_UPSAMPLE_VEC_IMPL(BGENTYPE##4, GENTYPE##4, UGENTYPE##4, GENSIZE)       \
  __CLC_UPSAMPLE_VEC_IMPL(BGENTYPE##8, GENTYPE##8, UGENTYPE##8, GENSIZE)       \
  __CLC_UPSAMPLE_VEC_IMPL(BGENTYPE##16, GENTYPE##16, UGENTYPE##16, GENSIZE)

#define __CLC_UPSAMPLE_TYPES()                                                 \
  __CLC_UPSAMPLE_IMPL(short, char, uchar, 8)                                   \
  __CLC_UPSAMPLE_IMPL(ushort, uchar, uchar, 8)                                 \
  __CLC_UPSAMPLE_IMPL(int, short, ushort, 16)                                  \
  __CLC_UPSAMPLE_IMPL(uint, ushort, ushort, 16)                                \
  __CLC_UPSAMPLE_IMPL(long, int, uint, 32)                                     \
  __CLC_UPSAMPLE_IMPL(ulong, uint, uint, 32)

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_IMPL
#undef __CLC_CONVERT_TY
