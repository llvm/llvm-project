#include <clc/clc.h>

//For all types EXCEPT long, which is implemented separately
#define __CLC_MUL_HI_IMPL(BGENTYPE, GENTYPE, GENSIZE) \
    _CLC_OVERLOAD _CLC_DEF GENTYPE mul_hi(GENTYPE x, GENTYPE y){ \
        return (GENTYPE)(((BGENTYPE)x * (BGENTYPE)y) >> GENSIZE); \
    } \

//FOIL-based long mul_hi
//
// Summary: Treat mul_hi(long x, long y) as:
// (a+b) * (c+d) where a and c are the high-order parts of x and y respectively
// and b and d are the low-order parts of x and y.
// Thinking back to algebra, we use FOIL to do the work.

_CLC_OVERLOAD _CLC_DEF long mul_hi(long x, long y){
    long f, o, i;
    ulong l;

    //Move the high/low halves of x/y into the lower 32-bits of variables so
    //that we can multiply them without worrying about overflow.
    long x_hi = x >> 32;
    long x_lo = x & UINT_MAX;
    long y_hi = y >> 32;
    long y_lo = y & UINT_MAX;

    //Multiply all of the components according to FOIL method
    f = x_hi * y_hi;
    o = x_hi * y_lo;
    i = x_lo * y_hi;
    l = x_lo * y_lo;

    //Now add the components back together in the following steps:
    //F: doesn't need to be modified
    //O/I: Need to be added together.
    //L: Shift right by 32-bits, then add into the sum of O and I
    //Once O/I/L are summed up, then shift the sum by 32-bits and add to F.
    //
    //We use hadd to give us a bit of extra precision for the intermediate sums
    //but as a result, we shift by 31 bits instead of 32
    return (long)(f + (hadd(o, (i + (long)((ulong)l>>32))) >> 31));
}

_CLC_OVERLOAD _CLC_DEF ulong mul_hi(ulong x, ulong y){
    ulong f, o, i;
    ulong l;

    //Move the high/low halves of x/y into the lower 32-bits of variables so
    //that we can multiply them without worrying about overflow.
    ulong x_hi = x >> 32;
    ulong x_lo = x & UINT_MAX;
    ulong y_hi = y >> 32;
    ulong y_lo = y & UINT_MAX;

    //Multiply all of the components according to FOIL method
    f = x_hi * y_hi;
    o = x_hi * y_lo;
    i = x_lo * y_hi;
    l = x_lo * y_lo;

    //Now add the components back together, taking care to respect the fact that:
    //F: doesn't need to be modified
    //O/I: Need to be added together.
    //L: Shift right by 32-bits, then add into the sum of O and I
    //Once O/I/L are summed up, then shift the sum by 32-bits and add to F.
    //
    //We use hadd to give us a bit of extra precision for the intermediate sums
    //but as a result, we shift by 31 bits instead of 32
    return (f + (hadd(o, (i + (l>>32))) >> 31));
}

#define __CLC_MUL_HI_VEC(GENTYPE) \
    _CLC_OVERLOAD _CLC_DEF GENTYPE##2 mul_hi(GENTYPE##2 x, GENTYPE##2 y){ \
        return (GENTYPE##2){mul_hi(x.s0, y.s0), mul_hi(x.s1, y.s1)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF GENTYPE##3 mul_hi(GENTYPE##3 x, GENTYPE##3 y){ \
        return (GENTYPE##3){mul_hi(x.s0, y.s0), mul_hi(x.s1, y.s1), mul_hi(x.s2, y.s2)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF GENTYPE##4 mul_hi(GENTYPE##4 x, GENTYPE##4 y){ \
        return (GENTYPE##4){mul_hi(x.lo, y.lo), mul_hi(x.hi, y.hi)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF GENTYPE##8 mul_hi(GENTYPE##8 x, GENTYPE##8 y){ \
        return (GENTYPE##8){mul_hi(x.lo, y.lo), mul_hi(x.hi, y.hi)}; \
    } \
    _CLC_OVERLOAD _CLC_DEF GENTYPE##16 mul_hi(GENTYPE##16 x, GENTYPE##16 y){ \
        return (GENTYPE##16){mul_hi(x.lo, y.lo), mul_hi(x.hi, y.hi)}; \
    } \

#define __CLC_MUL_HI_DEC_IMPL(BTYPE, TYPE, BITS) \
    __CLC_MUL_HI_IMPL(BTYPE, TYPE, BITS) \
    __CLC_MUL_HI_VEC(TYPE)

#define __CLC_MUL_HI_TYPES() \
    __CLC_MUL_HI_DEC_IMPL(short, char, 8) \
    __CLC_MUL_HI_DEC_IMPL(ushort, uchar, 8) \
    __CLC_MUL_HI_DEC_IMPL(int, short, 16) \
    __CLC_MUL_HI_DEC_IMPL(uint, ushort, 16) \
    __CLC_MUL_HI_DEC_IMPL(long, int, 32) \
    __CLC_MUL_HI_DEC_IMPL(ulong, uint, 32) \
    __CLC_MUL_HI_VEC(long) \
    __CLC_MUL_HI_VEC(ulong)

__CLC_MUL_HI_TYPES()

#undef __CLC_MUL_HI_TYPES
#undef __CLC_MUL_HI_DEC_IMPL
#undef __CLC_MUL_HI_IMPL
#undef __CLC_MUL_HI_VEC
#undef __CLC_B32
