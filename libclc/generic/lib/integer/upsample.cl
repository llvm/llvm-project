#include <clc/clc.h>
#include <clc/integer/clc_upsample.h>

#define __CLC_UPSAMPLE_IMPL(BGENTYPE, GENTYPE, UGENTYPE)                       \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE upsample(GENTYPE hi, UGENTYPE lo) {          \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##2 upsample(GENTYPE##2 hi, UGENTYPE##2 lo) { \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##3 upsample(GENTYPE##3 hi, UGENTYPE##3 lo) { \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##4 upsample(GENTYPE##4 hi, UGENTYPE##4 lo) { \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##8 upsample(GENTYPE##8 hi, UGENTYPE##8 lo) { \
    return __clc_upsample(hi, lo);                                             \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF BGENTYPE##16 upsample(GENTYPE##16 hi,                 \
                                               UGENTYPE##16 lo) {              \
    return __clc_upsample(hi, lo);                                             \
  }

#define __CLC_UPSAMPLE_TYPES()                                                 \
  __CLC_UPSAMPLE_IMPL(short, char, uchar)                                      \
  __CLC_UPSAMPLE_IMPL(ushort, uchar, uchar)                                    \
  __CLC_UPSAMPLE_IMPL(int, short, ushort)                                      \
  __CLC_UPSAMPLE_IMPL(uint, ushort, ushort)                                    \
  __CLC_UPSAMPLE_IMPL(long, int, uint)                                         \
  __CLC_UPSAMPLE_IMPL(ulong, uint, uint)

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_IMPL
