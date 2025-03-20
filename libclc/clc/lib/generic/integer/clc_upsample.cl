#include <clc/clc_convert.h>
#include <clc/internal/clc.h>

#define __CLC_UPSAMPLE_IMPL(BGENTYPE, GENTYPE, UGENTYPE, GENSIZE)              \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE __clc_upsample(GENTYPE hi, UGENTYPE lo) {    \
    BGENTYPE large_hi = __clc_convert_##BGENTYPE(hi);                          \
    BGENTYPE large_lo = __clc_convert_##BGENTYPE(lo);                          \
    return (large_hi << (BGENTYPE)GENSIZE) | large_lo;                         \
  }

#define __CLC_UPSAMPLE_IMPL_ALL_TYS(BGENTYPE, GENTYPE, UGENTYPE, GENSIZE)      \
  __CLC_UPSAMPLE_IMPL(BGENTYPE, GENTYPE, UGENTYPE, GENSIZE)                    \
  __CLC_UPSAMPLE_IMPL(BGENTYPE##2, GENTYPE##2, UGENTYPE##2, GENSIZE)           \
  __CLC_UPSAMPLE_IMPL(BGENTYPE##3, GENTYPE##3, UGENTYPE##3, GENSIZE)           \
  __CLC_UPSAMPLE_IMPL(BGENTYPE##4, GENTYPE##4, UGENTYPE##4, GENSIZE)           \
  __CLC_UPSAMPLE_IMPL(BGENTYPE##8, GENTYPE##8, UGENTYPE##8, GENSIZE)           \
  __CLC_UPSAMPLE_IMPL(BGENTYPE##16, GENTYPE##16, UGENTYPE##16, GENSIZE)

#define __CLC_UPSAMPLE_TYPES()                                                 \
  __CLC_UPSAMPLE_IMPL_ALL_TYS(short, char, uchar, 8)                           \
  __CLC_UPSAMPLE_IMPL_ALL_TYS(ushort, uchar, uchar, 8)                         \
  __CLC_UPSAMPLE_IMPL_ALL_TYS(int, short, ushort, 16)                          \
  __CLC_UPSAMPLE_IMPL_ALL_TYS(uint, ushort, ushort, 16)                        \
  __CLC_UPSAMPLE_IMPL_ALL_TYS(long, int, uint, 32)                             \
  __CLC_UPSAMPLE_IMPL_ALL_TYS(ulong, uint, uint, 32)

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_IMPL_ALL_TYS
#undef __CLC_UPSAMPLE_IMPL
