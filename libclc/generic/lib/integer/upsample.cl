#include <clc/clc.h>

#define __CLC_UPSAMPLE_IMPL(BGENTYPE, GENTYPE, UGENTYPE, GENSIZE) \
    _CLC_OVERLOAD _CLC_DEF BGENTYPE upsample(GENTYPE hi, UGENTYPE lo){ \
        return ((BGENTYPE)hi << GENSIZE) | lo; \
    } \
    _CLC_OVERLOAD _CLC_DEF BGENTYPE##2 upsample(GENTYPE##2 hi, UGENTYPE##2 lo){ \
        return (BGENTYPE##2){upsample(hi.s0, lo.s0), upsample(hi.s1, lo.s1)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF BGENTYPE##3 upsample(GENTYPE##3 hi, UGENTYPE##3 lo){ \
        return (BGENTYPE##3){upsample(hi.s0, lo.s0), upsample(hi.s1, lo.s1), upsample(hi.s2, lo.s2)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF BGENTYPE##4 upsample(GENTYPE##4 hi, UGENTYPE##4 lo){ \
        return (BGENTYPE##4){upsample(hi.lo, lo.lo), upsample(hi.hi, lo.hi)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF BGENTYPE##8 upsample(GENTYPE##8 hi, UGENTYPE##8 lo){ \
        return (BGENTYPE##8){upsample(hi.lo, lo.lo), upsample(hi.hi, lo.hi)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF BGENTYPE##16 upsample(GENTYPE##16 hi, UGENTYPE##16 lo){ \
        return (BGENTYPE##16){upsample(hi.lo, lo.lo), upsample(hi.hi, lo.hi)}; \
    } \

#define __CLC_UPSAMPLE_TYPES() \
    __CLC_UPSAMPLE_IMPL(short, char, uchar, 8) \
    __CLC_UPSAMPLE_IMPL(ushort, uchar, uchar, 8) \
    __CLC_UPSAMPLE_IMPL(int, short, ushort, 16) \
    __CLC_UPSAMPLE_IMPL(uint, ushort, ushort, 16) \
    __CLC_UPSAMPLE_IMPL(long, int, uint, 32) \
    __CLC_UPSAMPLE_IMPL(ulong, uint, uint, 32) \

__CLC_UPSAMPLE_TYPES()

#undef __CLC_UPSAMPLE_TYPES
#undef __CLC_UPSAMPLE_IMPL
