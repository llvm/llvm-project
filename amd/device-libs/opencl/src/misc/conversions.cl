/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ATTR __attribute__((overloadable, const))
#define IATTR __attribute__((const))
#define AATTR(S) __attribute__((overloadable, const, alias(S)))

#define _C(A,B) A##B
#define C(A,B) _C(A,B)


#if !defined USE_CLP
#define NOPN(N,TO,TI,S,R) ATTR TO##N convert_##TO##N##S##R(TO##N x) { return x; }

#define NOP(TO,TI,S,R) \
    NOPN(16,TO,TI,S,R) \
    NOPN(8,TO,TI,S,R) \
    NOPN(4,TO,TI,S,R) \
    NOPN(3,TO,TI,S,R) \
    NOPN(2,TO,TI,S,R) \
    NOPN(,TO,TI,S,R)

#define XLIST x
#define XLIST2 x.s0, x.s1
#define XLIST3 XLIST2, x.s2
#define XLIST4 XLIST3, x.s3
#define XLIST8 XLIST4, x.s4, x.s5, x.s6, x.s7
#define XLIST16 XLIST8, x.s8, x.s9, x.sa, x.sb, x.sc, x.sd, x.se, x.sf

#define YLIST y
#define YLIST2 y.s0, y.s1
#define YLIST3 YLIST2, y.s2
#define YLIST4 YLIST3, y.s3
#define YLIST8 YLIST4, y.s4, y.s5, y.s6, y.s7
#define YLIST16 YLIST8, y.s8, y.s9, y.sa, y.sb, y.sc, y.sd, y.se, y.sf

#define CASTN(N,TO,TI,S,R)  ATTR TO##N convert_##TO##N##S##R(TI##N x)  {  return (TO##N)(XLIST##N); }

#define CAST(TO,TI,S,R) \
    CASTN(16,TO,TI,S,R) \
    CASTN(8,TO,TI,S,R) \
    CASTN(4,TO,TI,S,R) \
    CASTN(3,TO,TI,S,R) \
    CASTN(2,TO,TI,S,R) \
    CASTN(,TO,TI,S,R)
#else
#define NOP(TO,TI,S,R)
#define CAST(TO,TI,S,R)
#endif

#define char_short_lb CHAR_MIN
#define char_short_ub CHAR_MAX
#define char_int_lb CHAR_MIN
#define char_int_ub CHAR_MAX
#define char_long_lb CHAR_MIN
#define char_long_ub CHAR_MAX
#define char_float_lb CHAR_MIN
#define char_float_ub CHAR_MAX
#define char_double_lb CHAR_MIN
#define char_double_ub CHAR_MAX
#define char_half_lb CHAR_MIN
#define char_half_ub CHAR_MAX

#define uchar_short_lb 0
#define uchar_short_ub UCHAR_MAX
#define uchar_int_lb 0
#define uchar_int_ub UCHAR_MAX
#define uchar_long_lb 0
#define uchar_long_ub UCHAR_MAX
#define uchar_float_lb 0
#define uchar_float_ub UCHAR_MAX
#define uchar_double_lb 0
#define uchar_double_ub UCHAR_MAX
#define uchar_half_lb 0
#define uchar_half_ub UCHAR_MAX

#define short_int_lb SHRT_MIN
#define short_int_ub SHRT_MAX
#define short_long_lb SHRT_MIN
#define short_long_ub SHRT_MAX
#define short_float_lb SHRT_MIN
#define short_float_ub SHRT_MAX
#define short_double_lb SHRT_MIN
#define short_double_ub SHRT_MAX
#define short_half_lb -HALF_MAX
#define short_half_ub HALF_MAX

#define ushort_int_lb 0
#define ushort_int_ub USHRT_MAX
#define ushort_long_lb 0
#define ushort_long_ub USHRT_MAX
#define ushort_float_lb 0
#define ushort_float_ub USHRT_MAX
#define ushort_double_lb 0
#define ushort_double_ub USHRT_MAX
#define ushort_half_lb 0
#define ushort_half_ub HALF_MAX

#define int_long_lb INT_MIN
#define int_long_ub INT_MAX
#define int_float_lb INT_MIN
#define int_float_ub 0x7fffff80
#define int_double_lb INT_MIN
#define int_double_ub INT_MAX
#define int_half_lb -HALF_MAX
#define int_half_ub HALF_MAX

#define uint_long_lb 0
#define uint_long_ub UINT_MAX
#define uint_float_lb 0
#define uint_float_ub 0xffffff00U
#define uint_double_lb 0
#define uint_double_ub UINT_MAX
#define uint_half_lb 0
#define uint_half_ub HALF_MAX

#define long_float_lb LONG_MIN
#define long_float_ub 0x7fffff8000000000L
#define long_double_lb LONG_MIN
#define long_double_ub 0x7ffffffffffffc00L
#define long_half_lb -HALF_MAX
#define long_half_ub HALF_MAX

#define ulong_float_lb 0
#define ulong_float_ub 0xffffff0000000000UL
#define ulong_double_lb 0
#define ulong_double_ub 0xfffffffffffff800UL
#define ulong_half_lb 0
#define ulong_half_ub HALF_MAX

#define char_minbnd CHAR_MAX
#define uchar_minbnd UCHAR_MAX
#define short_minbnd SHRT_MAX
#define ushort_minbnd USHRT_MAX
#define int_minbnd INT_MAX
#define uint_minbnd UINT_MAX
#define long_minbnd LONG_MAX
#define ulong_minbnd ULONG_MAX

#define char_maxbnd CHAR_MIN
#define uchar_maxbnd 0
#define short_maxbnd SHRT_MIN
#define ushort_maxbnd 0
#define int_maxbnd INT_MIN
#define uint_maxbnd 0
#define long_maxbnd LONG_MIN
#define ulong_maxbnd 0

#define MMN(F,N,TO,TI,S,R) \
ATTR TO##N \
convert_##TO##N##S##R(TI##N x) \
{ \
    return convert_##TO##N(F(x, (TI##N) TO##_##F##bnd)); \
}

#define MIN(TO,TI,S,R) \
    MMN(min,16,TO,TI,S,R) \
    MMN(min,8,TO,TI,S,R) \
    MMN(min,4,TO,TI,S,R) \
    MMN(min,3,TO,TI,S,R) \
    MMN(min,2,TO,TI,S,R) \
    MMN(min,,TO,TI,S,R)

#define MAX(TO,TI,S,R) \
    MMN(max,16,TO,TI,S,R) \
    MMN(max,8,TO,TI,S,R) \
    MMN(max,4,TO,TI,S,R) \
    MMN(max,3,TO,TI,S,R) \
    MMN(max,2,TO,TI,S,R) \
    MMN(max,,TO,TI,S,R)

#define CLAMPN(N,TO,TI,S,R) \
ATTR TO##N \
convert_##TO##N##S##R(TI##N x) \
{ \
    return convert_##TO##N(min(max(x, (TI##N) TO##_##TI##_lb), (TI##N) TO##_##TI##_ub)); \
}

#define CLAMP(TO,TI,S,R) \
    CLAMPN(16,TO,TI,S,R) \
    CLAMPN(8,TO,TI,S,R) \
    CLAMPN(4,TO,TI,S,R) \
    CLAMPN(3,TO,TI,S,R) \
    CLAMPN(2,TO,TI,S,R) \
    CLAMPN(,TO,TI,S,R)

#define F2IEN(E,N,TO,TI,S,R) \
ATTR TO##N \
convert_##TO##N##S##R(TI##N x) \
{ \
    return convert_##TO##N##_sat##E(x); \
}

#define F2IE(E,TO,TI,S,R) \
    F2IEN(E,16,TO,TI,S,R) \
    F2IEN(E,8,TO,TI,S,R) \
    F2IEN(E,4,TO,TI,S,R) \
    F2IEN(E,3,TO,TI,S,R) \
    F2IEN(E,2,TO,TI,S,R) \
    F2IEN(E,,TO,TI,S,R)

#define EF2I(TO,TI,S,R) F2IE(_rte,TO,TI,S,R)
#define NF2I(TO,TI,S,R) F2IE(_rtn,TO,TI,S,R)
#define PF2I(TO,TI,S,R) F2IE(_rtp,TO,TI,S,R)
#define ZF2I(TO,TI,S,R) F2IE(_rtz,TO,TI,S,R)

#define CLAMPFN(F,N,TO,TI,S,R) \
ATTR TO##N \
convert_##TO##N##S##R(TI##N x) \
{ \
    x = min(max(F(x), (TI##N) TO##_##TI##_lb), (TI##N) TO##_##TI##_ub); \
    return (TO##N)(XLIST##N); \
}

#define CLAMPF(F,TO,TI,S,R) \
    CLAMPFN(F,16,TO,TI,S,R) \
    CLAMPFN(F,8,TO,TI,S,R) \
    CLAMPFN(F,4,TO,TI,S,R) \
    CLAMPFN(F,3,TO,TI,S,R) \
    CLAMPFN(F,2,TO,TI,S,R) \
    CLAMPFN(F,,TO,TI,S,R)

#define ECLAMP(TO,TI,S,R) CLAMPF(rint,TO,TI,S,R)
#define NCLAMP(TO,TI,S,R) CLAMPF(floor,TO,TI,S,R)
#define PCLAMP(TO,TI,S,R) CLAMPF(ceil,TO,TI,S,R)
#define ZCLAMP(TO,TI,S,R) CLAMPF(,TO,TI,S,R)

#define SEL_(A,B,C) C ? B : A
#define SEL_2(A,B,C) select(A,B,C)
#define SEL_3(A,B,C) select(A,B,C)
#define SEL_4(A,B,C) select(A,B,C)
#define SEL_8(A,B,C) select(A,B,C)
#define SEL_16(A,B,C) select(A,B,C)

#define nou_short short
#define nou_ushort short
#define nou_int int
#define nou_uint int
#define nou_long long
#define nou_ulong long

#define CMP(N,TO,TI,X,OP,B) \
    C(convert_,C(nou_##TO, N))(X OP (TI##N) TO##_##TI##_##B)

#define CMP_(TO,TI,X,OP,B) (X OP (TI) TO##_##TI##_##B)
#define CMP_2(TO,TI,X,OP,B) CMP(2,TO,TI,X,OP,B)
#define CMP_3(TO,TI,X,OP,B) CMP(3,TO,TI,X,OP,B)
#define CMP_4(TO,TI,X,OP,B) CMP(4,TO,TI,X,OP,B)
#define CMP_8(TO,TI,X,OP,B) CMP(8,TO,TI,X,OP,B)
#define CMP_16(TO,TI,X,OP,B) CMP(16,TO,TI,X,OP,B)

#define CLAMP2FN(F,N,TO,TI,S,R) \
ATTR TO##N \
convert_##TO##N##S##R(TI##N x) \
{ \
    TI##N y = min(max(F(x), (TI##N) TO##_##TI##_lb), (TI##N) TO##_##TI##_ub); \
    TO##N z = (TO##N)(YLIST##N); \
    z = SEL_##N(z, (TO##N) TO##_minbnd, CMP_##N(TO,TI,x,>,ub)); \
    return SEL_##N(z, (TO##N) TO##_maxbnd, CMP_##N(TO,TI,x,<,lb)); \
}

#define CLAMP2F(F,TO,TI,S,R) \
    CLAMP2FN(F,16,TO,TI,S,R) \
    CLAMP2FN(F,8,TO,TI,S,R) \
    CLAMP2FN(F,4,TO,TI,S,R) \
    CLAMP2FN(F,3,TO,TI,S,R) \
    CLAMP2FN(F,2,TO,TI,S,R) \
    CLAMP2FN(F,,TO,TI,S,R)

#define ECLAMP2(TO,TI,S,R) CLAMP2F(rint,TO,TI,S,R)
#define NCLAMP2(TO,TI,S,R) CLAMP2F(floor,TO,TI,S,R)
#define PCLAMP2(TO,TI,S,R) CLAMP2F(ceil,TO,TI,S,R)
#define ZCLAMP2(TO,TI,S,R) CLAMP2F(,TO,TI,S,R)

#define EXPAND2(TO,TI,S,R) \
ATTR TO##2 \
convert_##TO##2##S##R(TI##2 x) \
{ \
    return (TO##2)(convert_##TO##S##R(x.lo), \
                   convert_##TO##S##R(x.hi)); \
}

#define EXPAND3(TO,TI,S,R) \
ATTR TO##3 \
convert_##TO##3##S##R(TI##3 x) \
{ \
    return (TO##3)(convert_##TO##2##S##R(x.s01), \
                   convert_##TO##S##R(x.s2)); \
}

#define EXPAND4(TO,TI,S,R) \
ATTR TO##4 \
convert_##TO##4##S##R(TI##4 x) \
{ \
    return (TO##4)(convert_##TO##2##S##R(x.lo), \
                   convert_##TO##2##S##R(x.hi)); \
}

#define EXPAND8(TO,TI,S,R) \
ATTR TO##8 \
convert_##TO##8##S##R(TI##8 x) \
{ \
    return (TO##8)(convert_##TO##4##S##R(x.lo), \
                   convert_##TO##4##S##R(x.hi)); \
}

#define EXPAND16(TO,TI,S,R) \
ATTR TO##16 \
convert_##TO##16##S##R(TI##16 x) \
{ \
    return (TO##16)(convert_##TO##8##S##R(x.lo), \
                    convert_##TO##8##S##R(x.hi)); \
}

#define EXPAND(TO,TI,S,R) \
    EXPAND16(TO,TI,S,R) \
    EXPAND8(TO,TI,S,R) \
    EXPAND4(TO,TI,S,R) \
    EXPAND3(TO,TI,S,R) \
    EXPAND2(TO,TI,S,R)

#define G_char_char(TO,TI,S,R)	        NOP(TO,TI,S,R)
#define G_char_sat_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_sat_rte_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_sat_rtn_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_sat_rtp_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_sat_rtz_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_rte_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_rtn_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_rtp_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_rtz_char(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_char_uchar(TO,TI,S,R)	        CAST(TO,TI,S,R)
#define G_char_sat_uchar(TO,TI,S,R)     MIN(TO,TI,S,R)
#define G_char_sat_rte_uchar(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_sat_rtn_uchar(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_sat_rtp_uchar(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_sat_rtz_uchar(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_short(TO,TI,S,R)	        CAST(TO,TI,S,R)
#define G_char_sat_short(TO,TI,S,R)     CLAMP(TO,TI,S,R)
#define G_char_sat_rte_short(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_char_sat_rtn_short(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_char_sat_rtp_short(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_char_sat_rtz_short(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_char_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_sat_ushort(TO,TI,S,R)    MIN(TO,TI,S,R)
#define G_char_sat_rte_ushort(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_char_sat_rtn_ushort(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_char_sat_rtp_ushort(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_char_sat_rtz_ushort(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_char_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_int(TO,TI,S,R)   	CAST(TO,TI,S,R)
#define G_char_sat_int(TO,TI,S,R)       CLAMP(TO,TI,S,R)
#define G_char_sat_rte_int(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_char_sat_rtn_int(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_char_sat_rtp_int(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_char_sat_rtz_int(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_char_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_uint(TO,TI,S,R)  	CAST(TO,TI,S,R)
#define G_char_sat_uint(TO,TI,S,R)      MIN(TO,TI,S,R)
#define G_char_sat_rte_uint(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_char_sat_rtn_uint(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_char_sat_rtp_uint(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_char_sat_rtz_uint(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_char_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_long(TO,TI,S,R)  	CAST(TO,TI,S,R)
#define G_char_sat_long(TO,TI,S,R)      CLAMP(TO,TI,S,R)
#define G_char_sat_rte_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_char_sat_rtn_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_char_sat_rtp_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_char_sat_rtz_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_char_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_ulong(TO,TI,S,R) 	CAST(TO,TI,S,R)
#define G_char_sat_ulong(TO,TI,S,R)     MIN(TO,TI,S,R)
#define G_char_sat_rte_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_sat_rtn_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_sat_rtp_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_sat_rtz_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_char_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_char_float(TO,TI,S,R) 	ZF2I(TO,TI,S,R)
#define G_char_sat_float(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_char_sat_rte_float(TO,TI,S,R) ECLAMP(TO,TI,S,R)
#define G_char_sat_rtn_float(TO,TI,S,R) NCLAMP(TO,TI,S,R)
#define G_char_sat_rtp_float(TO,TI,S,R) PCLAMP(TO,TI,S,R)
#define G_char_sat_rtz_float(TO,TI,S,R) ZCLAMP(TO,TI,S,R)
#define G_char_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_char_rtn_float(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_char_rtp_float(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_char_rtz_float(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_char_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_char_sat_double(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_char_sat_rte_double(TO,TI,S,R)        ECLAMP(TO,TI,S,R)
#define G_char_sat_rtn_double(TO,TI,S,R)        NCLAMP(TO,TI,S,R)
#define G_char_sat_rtp_double(TO,TI,S,R)        PCLAMP(TO,TI,S,R)
#define G_char_sat_rtz_double(TO,TI,S,R)        ZCLAMP(TO,TI,S,R)
#define G_char_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_char_rtn_double(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_char_rtp_double(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_char_rtz_double(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_char_half(TO,TI,S,R)  	ZF2I(TO,TI,S,R)
#define G_char_sat_half(TO,TI,S,R)      ZF2I(TO,TI,S,R)
#define G_char_sat_rte_half(TO,TI,S,R)  ECLAMP(TO,TI,S,R)
#define G_char_sat_rtn_half(TO,TI,S,R)  NCLAMP(TO,TI,S,R)
#define G_char_sat_rtp_half(TO,TI,S,R)  PCLAMP(TO,TI,S,R)
#define G_char_sat_rtz_half(TO,TI,S,R)  ZCLAMP(TO,TI,S,R)
#define G_char_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_char_rtn_half(TO,TI,S,R)      NF2I(TO,TI,S,R)
#define G_char_rtp_half(TO,TI,S,R)      PF2I(TO,TI,S,R)
#define G_char_rtz_half(TO,TI,S,R)      ZF2I(TO,TI,S,R)

#define G_uchar_char(TO,TI,S,R) 	CAST(TO,TI,S,R)
#define G_uchar_sat_char(TO,TI,S,R)     MAX(TO,TI,S,R)
#define G_uchar_sat_rte_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uchar_sat_rtn_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uchar_sat_rtp_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uchar_sat_rtz_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uchar_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_sat_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_sat_rte_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_sat_rtn_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_sat_rtp_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_sat_rtz_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_rte_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_rtn_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_rtp_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_rtz_uchar(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uchar_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_sat_short(TO,TI,S,R)    CLAMP(TO,TI,S,R)
#define G_uchar_sat_rte_short(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtn_short(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtp_short(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtz_short(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_uchar_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_sat_ushort(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_uchar_sat_rte_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_uchar_sat_rtn_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_uchar_sat_rtp_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_uchar_sat_rtz_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_uchar_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_sat_int(TO,TI,S,R)      CLAMP(TO,TI,S,R)
#define G_uchar_sat_rte_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtn_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtp_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtz_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uchar_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_sat_uint(TO,TI,S,R)     MIN(TO,TI,S,R)
#define G_uchar_sat_rte_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uchar_sat_rtn_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uchar_sat_rtp_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uchar_sat_rtz_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uchar_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_sat_long(TO,TI,S,R)     CLAMP(TO,TI,S,R)
#define G_uchar_sat_rte_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtn_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtp_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_uchar_sat_rtz_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_uchar_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_sat_ulong(TO,TI,S,R)    MIN(TO,TI,S,R)
#define G_uchar_sat_rte_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_uchar_sat_rtn_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_uchar_sat_rtp_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_uchar_sat_rtz_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_uchar_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uchar_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_uchar_sat_float(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_uchar_sat_rte_float(TO,TI,S,R)        ECLAMP(TO,TI,S,R)
#define G_uchar_sat_rtn_float(TO,TI,S,R)        NCLAMP(TO,TI,S,R)
#define G_uchar_sat_rtp_float(TO,TI,S,R)        PCLAMP(TO,TI,S,R)
#define G_uchar_sat_rtz_float(TO,TI,S,R)        ZCLAMP(TO,TI,S,R)
#define G_uchar_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_uchar_rtn_float(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_uchar_rtp_float(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_uchar_rtz_float(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_uchar_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_uchar_sat_double(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_uchar_sat_rte_double(TO,TI,S,R)       ECLAMP(TO,TI,S,R)
#define G_uchar_sat_rtn_double(TO,TI,S,R)       NCLAMP(TO,TI,S,R)
#define G_uchar_sat_rtp_double(TO,TI,S,R)       PCLAMP(TO,TI,S,R)
#define G_uchar_sat_rtz_double(TO,TI,S,R)       ZCLAMP(TO,TI,S,R)
#define G_uchar_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_uchar_rtn_double(TO,TI,S,R)   NF2I(TO,TI,S,R)
#define G_uchar_rtp_double(TO,TI,S,R)   PF2I(TO,TI,S,R)
#define G_uchar_rtz_double(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_uchar_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_uchar_sat_half(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_uchar_sat_rte_half(TO,TI,S,R) ECLAMP(TO,TI,S,R)
#define G_uchar_sat_rtn_half(TO,TI,S,R) NCLAMP(TO,TI,S,R)
#define G_uchar_sat_rtp_half(TO,TI,S,R) PCLAMP(TO,TI,S,R)
#define G_uchar_sat_rtz_half(TO,TI,S,R) ZCLAMP(TO,TI,S,R)
#define G_uchar_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_uchar_rtn_half(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_uchar_rtp_half(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_uchar_rtz_half(TO,TI,S,R)     ZF2I(TO,TI,S,R)

#define G_short_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_sat_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_sat_rte_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_sat_rtn_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_sat_rtp_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_sat_rtz_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_rte_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_rtn_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_rtp_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_rtz_short(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_short_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_ushort(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_short_sat_rte_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_short_sat_rtn_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_short_sat_rtp_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_short_sat_rtz_ushort(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_short_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_int(TO,TI,S,R)      CLAMP(TO,TI,S,R)
#define G_short_sat_rte_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_short_sat_rtn_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_short_sat_rtp_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_short_sat_rtz_int(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_short_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_uint(TO,TI,S,R)     MIN(TO,TI,S,R)
#define G_short_sat_rte_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_short_sat_rtn_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_short_sat_rtp_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_short_sat_rtz_uint(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_short_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_long(TO,TI,S,R)     CLAMP(TO,TI,S,R)
#define G_short_sat_rte_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_short_sat_rtn_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_short_sat_rtp_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_short_sat_rtz_long(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_short_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_sat_ulong(TO,TI,S,R)    MIN(TO,TI,S,R)
#define G_short_sat_rte_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_short_sat_rtn_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_short_sat_rtp_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_short_sat_rtz_ulong(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_short_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_short_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_short_sat_float(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_short_sat_rte_float(TO,TI,S,R)        ECLAMP(TO,TI,S,R)
#define G_short_sat_rtn_float(TO,TI,S,R)        NCLAMP(TO,TI,S,R)
#define G_short_sat_rtp_float(TO,TI,S,R)        PCLAMP(TO,TI,S,R)
#define G_short_sat_rtz_float(TO,TI,S,R)        ZCLAMP(TO,TI,S,R)
#define G_short_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_short_rtn_float(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_short_rtp_float(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_short_rtz_float(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_short_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_short_sat_double(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_short_sat_rte_double(TO,TI,S,R)       ECLAMP(TO,TI,S,R)
#define G_short_sat_rtn_double(TO,TI,S,R)       NCLAMP(TO,TI,S,R)
#define G_short_sat_rtp_double(TO,TI,S,R)       PCLAMP(TO,TI,S,R)
#define G_short_sat_rtz_double(TO,TI,S,R)       ZCLAMP(TO,TI,S,R)
#define G_short_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_short_rtn_double(TO,TI,S,R)   NF2I(TO,TI,S,R)
#define G_short_rtp_double(TO,TI,S,R)   PF2I(TO,TI,S,R)
#define G_short_rtz_double(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_short_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_short_sat_half(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_short_sat_rte_half(TO,TI,S,R) ECLAMP2(TO,TI,S,R)
#define G_short_sat_rtn_half(TO,TI,S,R) NCLAMP2(TO,TI,S,R)
#define G_short_sat_rtp_half(TO,TI,S,R) PCLAMP2(TO,TI,S,R)
#define G_short_sat_rtz_half(TO,TI,S,R) ZCLAMP2(TO,TI,S,R)
#define G_short_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_short_rtn_half(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_short_rtp_half(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_short_rtz_half(TO,TI,S,R)     ZF2I(TO,TI,S,R)

#define G_ushort_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_char(TO,TI,S,R)    MAX(TO,TI,S,R)
#define G_ushort_sat_rte_char(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ushort_sat_rtn_char(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ushort_sat_rtp_char(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ushort_sat_rtz_char(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ushort_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_short(TO,TI,S,R)   MAX(TO,TI,S,R)
#define G_ushort_sat_rte_short(TO,TI,S,R)       MAX(TO,TI,S,R)
#define G_ushort_sat_rtn_short(TO,TI,S,R)       MAX(TO,TI,S,R)
#define G_ushort_sat_rtp_short(TO,TI,S,R)       MAX(TO,TI,S,R)
#define G_ushort_sat_rtz_short(TO,TI,S,R)       MAX(TO,TI,S,R)
#define G_ushort_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_sat_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_sat_rte_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_sat_rtn_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_sat_rtp_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_sat_rtz_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_rte_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_rtn_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_rtp_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_rtz_ushort(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ushort_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_int(TO,TI,S,R)     CLAMP(TO,TI,S,R)
#define G_ushort_sat_rte_int(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_ushort_sat_rtn_int(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_ushort_sat_rtp_int(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_ushort_sat_rtz_int(TO,TI,S,R) CLAMP(TO,TI,S,R)
#define G_ushort_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_uint(TO,TI,S,R)    MIN(TO,TI,S,R)
#define G_ushort_sat_rte_uint(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_ushort_sat_rtn_uint(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_ushort_sat_rtp_uint(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_ushort_sat_rtz_uint(TO,TI,S,R)        MIN(TO,TI,S,R)
#define G_ushort_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_long(TO,TI,S,R)    CLAMP(TO,TI,S,R)
#define G_ushort_sat_rte_long(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_ushort_sat_rtn_long(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_ushort_sat_rtp_long(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_ushort_sat_rtz_long(TO,TI,S,R)        CLAMP(TO,TI,S,R)
#define G_ushort_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_sat_ulong(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_ushort_sat_rte_ulong(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_ushort_sat_rtn_ulong(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_ushort_sat_rtp_ulong(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_ushort_sat_rtz_ulong(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_ushort_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ushort_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_ushort_sat_float(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_ushort_sat_rte_float(TO,TI,S,R)       ECLAMP(TO,TI,S,R)
#define G_ushort_sat_rtn_float(TO,TI,S,R)       NCLAMP(TO,TI,S,R)
#define G_ushort_sat_rtp_float(TO,TI,S,R)       PCLAMP(TO,TI,S,R)
#define G_ushort_sat_rtz_float(TO,TI,S,R)       ZCLAMP(TO,TI,S,R)
#define G_ushort_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_ushort_rtn_float(TO,TI,S,R)   NF2I(TO,TI,S,R)
#define G_ushort_rtp_float(TO,TI,S,R)   PF2I(TO,TI,S,R)
#define G_ushort_rtz_float(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_ushort_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_ushort_sat_double(TO,TI,S,R)  ZF2I(TO,TI,S,R)
#define G_ushort_sat_rte_double(TO,TI,S,R)      ECLAMP(TO,TI,S,R)
#define G_ushort_sat_rtn_double(TO,TI,S,R)      NCLAMP(TO,TI,S,R)
#define G_ushort_sat_rtp_double(TO,TI,S,R)      PCLAMP(TO,TI,S,R)
#define G_ushort_sat_rtz_double(TO,TI,S,R)      ZCLAMP(TO,TI,S,R)
#define G_ushort_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_ushort_rtn_double(TO,TI,S,R)  NF2I(TO,TI,S,R)
#define G_ushort_rtp_double(TO,TI,S,R)  PF2I(TO,TI,S,R)
#define G_ushort_rtz_double(TO,TI,S,R)  ZF2I(TO,TI,S,R)
#define G_ushort_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_ushort_sat_half(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_ushort_sat_rte_half(TO,TI,S,R)        ECLAMP2(TO,TI,S,R)
#define G_ushort_sat_rtn_half(TO,TI,S,R)        NCLAMP2(TO,TI,S,R)
#define G_ushort_sat_rtp_half(TO,TI,S,R)        PCLAMP2(TO,TI,S,R)
#define G_ushort_sat_rtz_half(TO,TI,S,R)        ZCLAMP2(TO,TI,S,R)
#define G_ushort_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_ushort_rtn_half(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_ushort_rtp_half(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_ushort_rtz_half(TO,TI,S,R)    ZF2I(TO,TI,S,R)

#define G_int_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_sat_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_sat_rte_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_sat_rtn_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_sat_rtp_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_sat_rtz_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_rte_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_rtn_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_rtp_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_rtz_int(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_int_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_uint(TO,TI,S,R)       MIN(TO,TI,S,R)
#define G_int_sat_rte_uint(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_int_sat_rtn_uint(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_int_sat_rtp_uint(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_int_sat_rtz_uint(TO,TI,S,R)   MIN(TO,TI,S,R)
#define G_int_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_long(TO,TI,S,R)       CLAMP(TO,TI,S,R)
#define G_int_sat_rte_long(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_int_sat_rtn_long(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_int_sat_rtp_long(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_int_sat_rtz_long(TO,TI,S,R)   CLAMP(TO,TI,S,R)
#define G_int_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_sat_ulong(TO,TI,S,R)      MIN(TO,TI,S,R)
#define G_int_sat_rte_ulong(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_int_sat_rtn_ulong(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_int_sat_rtp_ulong(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_int_sat_rtz_ulong(TO,TI,S,R)  MIN(TO,TI,S,R)
#define G_int_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_int_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_int_sat_float(TO,TI,S,R)      ZF2I(TO,TI,S,R)
#define G_int_sat_rte_float(TO,TI,S,R)  ECLAMP2(TO,TI,S,R)
#define G_int_sat_rtn_float(TO,TI,S,R)  NCLAMP2(TO,TI,S,R)
#define G_int_sat_rtp_float(TO,TI,S,R)  PCLAMP2(TO,TI,S,R)
#define G_int_sat_rtz_float(TO,TI,S,R)  ZCLAMP2(TO,TI,S,R)
#define G_int_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_int_rtn_float(TO,TI,S,R)      NF2I(TO,TI,S,R)
#define G_int_rtp_float(TO,TI,S,R)      PF2I(TO,TI,S,R)
#define G_int_rtz_float(TO,TI,S,R)      ZF2I(TO,TI,S,R)
#define G_int_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_int_sat_double(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_int_sat_rte_double(TO,TI,S,R) ECLAMP(TO,TI,S,R)
#define G_int_sat_rtn_double(TO,TI,S,R) NCLAMP(TO,TI,S,R)
#define G_int_sat_rtp_double(TO,TI,S,R) PCLAMP(TO,TI,S,R)
#define G_int_sat_rtz_double(TO,TI,S,R) ZCLAMP(TO,TI,S,R)
#define G_int_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_int_rtn_double(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_int_rtp_double(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_int_rtz_double(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_int_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_int_sat_half(TO,TI,S,R)       ZF2I(TO,TI,S,R)
#define G_int_sat_rte_half(TO,TI,S,R)   ECLAMP2(TO,TI,S,R)
#define G_int_sat_rtn_half(TO,TI,S,R)   NCLAMP2(TO,TI,S,R)
#define G_int_sat_rtp_half(TO,TI,S,R)   PCLAMP2(TO,TI,S,R)
#define G_int_sat_rtz_half(TO,TI,S,R)   ZCLAMP2(TO,TI,S,R)
#define G_int_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_int_rtn_half(TO,TI,S,R)       NF2I(TO,TI,S,R)
#define G_int_rtp_half(TO,TI,S,R)       PF2I(TO,TI,S,R)
#define G_int_rtz_half(TO,TI,S,R)       ZF2I(TO,TI,S,R)

#define G_uint_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_char(TO,TI,S,R)      MAX(TO,TI,S,R)
#define G_uint_sat_rte_char(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_uint_sat_rtn_char(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_uint_sat_rtp_char(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_uint_sat_rtz_char(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_uint_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_short(TO,TI,S,R)     MAX(TO,TI,S,R)
#define G_uint_sat_rte_short(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uint_sat_rtn_short(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uint_sat_rtp_short(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uint_sat_rtz_short(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_uint_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_int(TO,TI,S,R)       MAX(TO,TI,S,R)
#define G_uint_sat_rte_int(TO,TI,S,R)   MAX(TO,TI,S,R)
#define G_uint_sat_rtn_int(TO,TI,S,R)   MAX(TO,TI,S,R)
#define G_uint_sat_rtp_int(TO,TI,S,R)   MAX(TO,TI,S,R)
#define G_uint_sat_rtz_int(TO,TI,S,R)   MAX(TO,TI,S,R)
#define G_uint_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_sat_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_sat_rte_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_sat_rtn_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_sat_rtp_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_sat_rtz_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_rte_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_rtn_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_rtp_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_rtz_uint(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_uint_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_long(TO,TI,S,R)      CLAMP(TO,TI,S,R)
#define G_uint_sat_rte_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uint_sat_rtn_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uint_sat_rtp_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uint_sat_rtz_long(TO,TI,S,R)  CLAMP(TO,TI,S,R)
#define G_uint_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_sat_ulong(TO,TI,S,R)     MIN(TO,TI,S,R)
#define G_uint_sat_rte_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uint_sat_rtn_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uint_sat_rtp_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uint_sat_rtz_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_uint_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_uint_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_uint_sat_float(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_uint_sat_rte_float(TO,TI,S,R) ECLAMP2(TO,TI,S,R)
#define G_uint_sat_rtn_float(TO,TI,S,R) NCLAMP2(TO,TI,S,R)
#define G_uint_sat_rtp_float(TO,TI,S,R) PCLAMP2(TO,TI,S,R)
#define G_uint_sat_rtz_float(TO,TI,S,R) ZCLAMP2(TO,TI,S,R)
#define G_uint_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_uint_rtn_float(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_uint_rtp_float(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_uint_rtz_float(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_uint_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_uint_sat_double(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_uint_sat_rte_double(TO,TI,S,R)        ECLAMP(TO,TI,S,R)
#define G_uint_sat_rtn_double(TO,TI,S,R)        NCLAMP(TO,TI,S,R)
#define G_uint_sat_rtp_double(TO,TI,S,R)        PCLAMP(TO,TI,S,R)
#define G_uint_sat_rtz_double(TO,TI,S,R)        ZCLAMP(TO,TI,S,R)
#define G_uint_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_uint_rtn_double(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_uint_rtp_double(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_uint_rtz_double(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_uint_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_uint_sat_half(TO,TI,S,R)      ZF2I(TO,TI,S,R)
#define G_uint_sat_rte_half(TO,TI,S,R)  ECLAMP2(TO,TI,S,R)
#define G_uint_sat_rtn_half(TO,TI,S,R)  NCLAMP2(TO,TI,S,R)
#define G_uint_sat_rtp_half(TO,TI,S,R)  PCLAMP2(TO,TI,S,R)
#define G_uint_sat_rtz_half(TO,TI,S,R)  ZCLAMP2(TO,TI,S,R)
#define G_uint_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_uint_rtn_half(TO,TI,S,R)      NF2I(TO,TI,S,R)
#define G_uint_rtp_half(TO,TI,S,R)      PF2I(TO,TI,S,R)
#define G_uint_rtz_half(TO,TI,S,R)      ZF2I(TO,TI,S,R)

#define G_long_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_sat_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_sat_rte_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_sat_rtn_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_sat_rtp_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_sat_rtz_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_rte_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_rtn_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_rtp_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_rtz_long(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_long_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_sat_ulong(TO,TI,S,R)     MIN(TO,TI,S,R)
#define G_long_sat_rte_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_long_sat_rtn_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_long_sat_rtp_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_long_sat_rtz_ulong(TO,TI,S,R) MIN(TO,TI,S,R)
#define G_long_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtn_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtp_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_rtz_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_long_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_long_sat_float(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_long_sat_rte_float(TO,TI,S,R) ECLAMP2(TO,TI,S,R)
#define G_long_sat_rtn_float(TO,TI,S,R) NCLAMP2(TO,TI,S,R)
#define G_long_sat_rtp_float(TO,TI,S,R) PCLAMP2(TO,TI,S,R)
#define G_long_sat_rtz_float(TO,TI,S,R) ZCLAMP2(TO,TI,S,R)
#define G_long_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_long_rtn_float(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_long_rtp_float(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_long_rtz_float(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_long_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_long_sat_double(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_long_sat_rte_double(TO,TI,S,R)        ECLAMP2(TO,TI,S,R)
#define G_long_sat_rtn_double(TO,TI,S,R)        NCLAMP2(TO,TI,S,R)
#define G_long_sat_rtp_double(TO,TI,S,R)        PCLAMP2(TO,TI,S,R)
#define G_long_sat_rtz_double(TO,TI,S,R)        ZCLAMP2(TO,TI,S,R)
#define G_long_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_long_rtn_double(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_long_rtp_double(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_long_rtz_double(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_long_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_long_sat_half(TO,TI,S,R)      ZF2I(TO,TI,S,R)
#define G_long_sat_rte_half(TO,TI,S,R)  ECLAMP2(TO,TI,S,R)
#define G_long_sat_rtn_half(TO,TI,S,R)  NCLAMP2(TO,TI,S,R)
#define G_long_sat_rtp_half(TO,TI,S,R)  PCLAMP2(TO,TI,S,R)
#define G_long_sat_rtz_half(TO,TI,S,R)  ZCLAMP2(TO,TI,S,R)
#define G_long_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_long_rtn_half(TO,TI,S,R)      NF2I(TO,TI,S,R)
#define G_long_rtp_half(TO,TI,S,R)      PF2I(TO,TI,S,R)
#define G_long_rtz_half(TO,TI,S,R)      ZF2I(TO,TI,S,R)

#define G_ulong_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_char(TO,TI,S,R)     MAX(TO,TI,S,R)
#define G_ulong_sat_rte_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_sat_rtn_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_sat_rtp_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_sat_rtz_char(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_short(TO,TI,S,R)    MAX(TO,TI,S,R)
#define G_ulong_sat_rte_short(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ulong_sat_rtn_short(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ulong_sat_rtp_short(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ulong_sat_rtz_short(TO,TI,S,R)        MAX(TO,TI,S,R)
#define G_ulong_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_int(TO,TI,S,R)      MAX(TO,TI,S,R)
#define G_ulong_sat_rte_int(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_ulong_sat_rtn_int(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_ulong_sat_rtp_int(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_ulong_sat_rtz_int(TO,TI,S,R)  MAX(TO,TI,S,R)
#define G_ulong_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_sat_long(TO,TI,S,R)     MAX(TO,TI,S,R)
#define G_ulong_sat_rte_long(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_sat_rtn_long(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_sat_rtp_long(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_sat_rtz_long(TO,TI,S,R) MAX(TO,TI,S,R)
#define G_ulong_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtn_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtp_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_rtz_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_ulong_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_sat_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_sat_rte_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_sat_rtn_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_sat_rtp_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_sat_rtz_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_rte_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_rtn_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_rtp_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_rtz_ulong(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_ulong_float(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_ulong_sat_float(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_ulong_sat_rte_float(TO,TI,S,R)        ECLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtn_float(TO,TI,S,R)        NCLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtp_float(TO,TI,S,R)        PCLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtz_float(TO,TI,S,R)        ZCLAMP2(TO,TI,S,R)
#define G_ulong_rte_float(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_ulong_rtn_float(TO,TI,S,R)    NF2I(TO,TI,S,R)
#define G_ulong_rtp_float(TO,TI,S,R)    PF2I(TO,TI,S,R)
#define G_ulong_rtz_float(TO,TI,S,R)    ZF2I(TO,TI,S,R)
#define G_ulong_double(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_ulong_sat_double(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_ulong_sat_rte_double(TO,TI,S,R)       ECLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtn_double(TO,TI,S,R)       NCLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtp_double(TO,TI,S,R)       PCLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtz_double(TO,TI,S,R)       ZCLAMP2(TO,TI,S,R)
#define G_ulong_rte_double(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_ulong_rtn_double(TO,TI,S,R)   NF2I(TO,TI,S,R)
#define G_ulong_rtp_double(TO,TI,S,R)   PF2I(TO,TI,S,R)
#define G_ulong_rtz_double(TO,TI,S,R)   ZF2I(TO,TI,S,R)
#define G_ulong_half(TO,TI,S,R)	ZF2I(TO,TI,S,R)
#define G_ulong_sat_half(TO,TI,S,R)     ZF2I(TO,TI,S,R)
#define G_ulong_sat_rte_half(TO,TI,S,R) ECLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtn_half(TO,TI,S,R) NCLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtp_half(TO,TI,S,R) PCLAMP2(TO,TI,S,R)
#define G_ulong_sat_rtz_half(TO,TI,S,R) ZCLAMP2(TO,TI,S,R)
#define G_ulong_rte_half(TO,TI,S,R)	EF2I(TO,TI,S,R)
#define G_ulong_rtn_half(TO,TI,S,R)     NF2I(TO,TI,S,R)
#define G_ulong_rtp_half(TO,TI,S,R)     PF2I(TO,TI,S,R)
#define G_ulong_rtz_half(TO,TI,S,R)     ZF2I(TO,TI,S,R)

#define G_float_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_char(TO,TI,S,R)
#define G_float_sat_rte_char(TO,TI,S,R)
#define G_float_sat_rtn_char(TO,TI,S,R)
#define G_float_sat_rtp_char(TO,TI,S,R)
#define G_float_sat_rtz_char(TO,TI,S,R)
#define G_float_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_uchar(TO,TI,S,R)
#define G_float_sat_rte_uchar(TO,TI,S,R)
#define G_float_sat_rtn_uchar(TO,TI,S,R)
#define G_float_sat_rtp_uchar(TO,TI,S,R)
#define G_float_sat_rtz_uchar(TO,TI,S,R)
#define G_float_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_short(TO,TI,S,R)
#define G_float_sat_rte_short(TO,TI,S,R)
#define G_float_sat_rtn_short(TO,TI,S,R)
#define G_float_sat_rtp_short(TO,TI,S,R)
#define G_float_sat_rtz_short(TO,TI,S,R)
#define G_float_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_ushort(TO,TI,S,R)
#define G_float_sat_rte_ushort(TO,TI,S,R)
#define G_float_sat_rtn_ushort(TO,TI,S,R)
#define G_float_sat_rtp_ushort(TO,TI,S,R)
#define G_float_sat_rtz_ushort(TO,TI,S,R)
#define G_float_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_int(TO,TI,S,R)
#define G_float_sat_rte_int(TO,TI,S,R)
#define G_float_sat_rtn_int(TO,TI,S,R)
#define G_float_sat_rtp_int(TO,TI,S,R)
#define G_float_sat_rtz_int(TO,TI,S,R)
#define G_float_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_int(TO,TI,S,R)      EXPAND(TO,TI,S,R)
#define G_float_rtp_int(TO,TI,S,R)      EXPAND(TO,TI,S,R)
#define G_float_rtz_int(TO,TI,S,R)      EXPAND(TO,TI,S,R)
#define G_float_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_uint(TO,TI,S,R)
#define G_float_sat_rte_uint(TO,TI,S,R)
#define G_float_sat_rtn_uint(TO,TI,S,R)
#define G_float_sat_rtp_uint(TO,TI,S,R)
#define G_float_sat_rtz_uint(TO,TI,S,R)
#define G_float_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_uint(TO,TI,S,R)     EXPAND(TO,TI,S,R)
#define G_float_rtp_uint(TO,TI,S,R)     EXPAND(TO,TI,S,R)
#define G_float_rtz_uint(TO,TI,S,R)     EXPAND(TO,TI,S,R)
#define G_float_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_long(TO,TI,S,R)
#define G_float_sat_rte_long(TO,TI,S,R)
#define G_float_sat_rtn_long(TO,TI,S,R)
#define G_float_sat_rtp_long(TO,TI,S,R)
#define G_float_sat_rtz_long(TO,TI,S,R)
#define G_float_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_long(TO,TI,S,R)     EXPAND(TO,TI,S,R)
#define G_float_rtp_long(TO,TI,S,R)     EXPAND(TO,TI,S,R)
#define G_float_rtz_long(TO,TI,S,R)     EXPAND(TO,TI,S,R)
#define G_float_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_ulong(TO,TI,S,R)
#define G_float_sat_rte_ulong(TO,TI,S,R)
#define G_float_sat_rtn_ulong(TO,TI,S,R)
#define G_float_sat_rtp_ulong(TO,TI,S,R)
#define G_float_sat_rtz_ulong(TO,TI,S,R)
#define G_float_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_ulong(TO,TI,S,R)    EXPAND(TO,TI,S,R)
#define G_float_rtp_ulong(TO,TI,S,R)    EXPAND(TO,TI,S,R)
#define G_float_rtz_ulong(TO,TI,S,R)    EXPAND(TO,TI,S,R)
#define G_float_float(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_float_sat_float(TO,TI,S,R)
#define G_float_sat_rte_float(TO,TI,S,R)
#define G_float_sat_rtn_float(TO,TI,S,R)
#define G_float_sat_rtp_float(TO,TI,S,R)
#define G_float_sat_rtz_float(TO,TI,S,R)
#define G_float_rte_float(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_float_rtn_float(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_float_rtp_float(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_float_rtz_float(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_float_double(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_double(TO,TI,S,R)
#define G_float_sat_rte_double(TO,TI,S,R)
#define G_float_sat_rtn_double(TO,TI,S,R)
#define G_float_sat_rtp_double(TO,TI,S,R)
#define G_float_sat_rtz_double(TO,TI,S,R)
#define G_float_rte_double(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_double(TO,TI,S,R)   EXPAND(TO,TI,S,R)
#define G_float_rtp_double(TO,TI,S,R)   EXPAND(TO,TI,S,R)
#define G_float_rtz_double(TO,TI,S,R)   EXPAND(TO,TI,S,R)
#define G_float_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_sat_half(TO,TI,S,R)
#define G_float_sat_rte_half(TO,TI,S,R)
#define G_float_sat_rtn_half(TO,TI,S,R)
#define G_float_sat_rtp_half(TO,TI,S,R)
#define G_float_sat_rtz_half(TO,TI,S,R)
#define G_float_rte_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtn_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtp_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_float_rtz_half(TO,TI,S,R)	CAST(TO,TI,S,R)

#define G_double_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_char(TO,TI,S,R)
#define G_double_sat_rte_char(TO,TI,S,R)
#define G_double_sat_rtn_char(TO,TI,S,R)
#define G_double_sat_rtp_char(TO,TI,S,R)
#define G_double_sat_rtz_char(TO,TI,S,R)
#define G_double_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_uchar(TO,TI,S,R)
#define G_double_sat_rte_uchar(TO,TI,S,R)
#define G_double_sat_rtn_uchar(TO,TI,S,R)
#define G_double_sat_rtp_uchar(TO,TI,S,R)
#define G_double_sat_rtz_uchar(TO,TI,S,R)
#define G_double_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_short(TO,TI,S,R)
#define G_double_sat_rte_short(TO,TI,S,R)
#define G_double_sat_rtn_short(TO,TI,S,R)
#define G_double_sat_rtp_short(TO,TI,S,R)
#define G_double_sat_rtz_short(TO,TI,S,R)
#define G_double_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_ushort(TO,TI,S,R)
#define G_double_sat_rte_ushort(TO,TI,S,R)
#define G_double_sat_rtn_ushort(TO,TI,S,R)
#define G_double_sat_rtp_ushort(TO,TI,S,R)
#define G_double_sat_rtz_ushort(TO,TI,S,R)
#define G_double_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_int(TO,TI,S,R)
#define G_double_sat_rte_int(TO,TI,S,R)
#define G_double_sat_rtn_int(TO,TI,S,R)
#define G_double_sat_rtp_int(TO,TI,S,R)
#define G_double_sat_rtz_int(TO,TI,S,R)
#define G_double_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_uint(TO,TI,S,R)
#define G_double_sat_rte_uint(TO,TI,S,R)
#define G_double_sat_rtn_uint(TO,TI,S,R)
#define G_double_sat_rtp_uint(TO,TI,S,R)
#define G_double_sat_rtz_uint(TO,TI,S,R)
#define G_double_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_long(TO,TI,S,R)
#define G_double_sat_rte_long(TO,TI,S,R)
#define G_double_sat_rtn_long(TO,TI,S,R)
#define G_double_sat_rtp_long(TO,TI,S,R)
#define G_double_sat_rtz_long(TO,TI,S,R)
#define G_double_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_long(TO,TI,S,R)    EXPAND(TO,TI,S,R)
#define G_double_rtp_long(TO,TI,S,R)    EXPAND(TO,TI,S,R)
#define G_double_rtz_long(TO,TI,S,R)    EXPAND(TO,TI,S,R)
#define G_double_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_ulong(TO,TI,S,R)
#define G_double_sat_rte_ulong(TO,TI,S,R)
#define G_double_sat_rtn_ulong(TO,TI,S,R)
#define G_double_sat_rtp_ulong(TO,TI,S,R)
#define G_double_sat_rtz_ulong(TO,TI,S,R)
#define G_double_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_ulong(TO,TI,S,R)   EXPAND(TO,TI,S,R)
#define G_double_rtp_ulong(TO,TI,S,R)   EXPAND(TO,TI,S,R)
#define G_double_rtz_ulong(TO,TI,S,R)   EXPAND(TO,TI,S,R)
#define G_double_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_float(TO,TI,S,R)
#define G_double_sat_rte_float(TO,TI,S,R)
#define G_double_sat_rtn_float(TO,TI,S,R)
#define G_double_sat_rtp_float(TO,TI,S,R)
#define G_double_sat_rtz_float(TO,TI,S,R)
#define G_double_rte_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_double(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_double_sat_double(TO,TI,S,R)
#define G_double_sat_rte_double(TO,TI,S,R)
#define G_double_sat_rtn_double(TO,TI,S,R)
#define G_double_sat_rtp_double(TO,TI,S,R)
#define G_double_sat_rtz_double(TO,TI,S,R)
#define G_double_rte_double(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_double_rtn_double(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_double_rtp_double(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_double_rtz_double(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_double_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_sat_half(TO,TI,S,R)
#define G_double_sat_rte_half(TO,TI,S,R)
#define G_double_sat_rtn_half(TO,TI,S,R)
#define G_double_sat_rtp_half(TO,TI,S,R)
#define G_double_sat_rtz_half(TO,TI,S,R)
#define G_double_rte_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtn_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtp_half(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_double_rtz_half(TO,TI,S,R)	CAST(TO,TI,S,R)

#define G_half_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_char(TO,TI,S,R)
#define G_half_sat_rte_char(TO,TI,S,R)
#define G_half_sat_rtn_char(TO,TI,S,R)
#define G_half_sat_rtp_char(TO,TI,S,R)
#define G_half_sat_rtz_char(TO,TI,S,R)
#define G_half_rte_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtp_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtz_char(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_uchar(TO,TI,S,R)
#define G_half_sat_rte_uchar(TO,TI,S,R)
#define G_half_sat_rtn_uchar(TO,TI,S,R)
#define G_half_sat_rtp_uchar(TO,TI,S,R)
#define G_half_sat_rtz_uchar(TO,TI,S,R)
#define G_half_rte_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtp_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtz_uchar(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_short(TO,TI,S,R)
#define G_half_sat_rte_short(TO,TI,S,R)
#define G_half_sat_rtn_short(TO,TI,S,R)
#define G_half_sat_rtp_short(TO,TI,S,R)
#define G_half_sat_rtz_short(TO,TI,S,R)
#define G_half_rte_short(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_short(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_rtp_short(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_rtz_short(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_ushort(TO,TI,S,R)
#define G_half_sat_rte_ushort(TO,TI,S,R)
#define G_half_sat_rtn_ushort(TO,TI,S,R)
#define G_half_sat_rtp_ushort(TO,TI,S,R)
#define G_half_sat_rtz_ushort(TO,TI,S,R)
#define G_half_rte_ushort(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_ushort(TO,TI,S,R)    EXPAND(TO,TI,R,S)
#define G_half_rtp_ushort(TO,TI,S,R)    EXPAND(TO,TI,R,S)
#define G_half_rtz_ushort(TO,TI,S,R)    EXPAND(TO,TI,R,S)
#define G_half_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_int(TO,TI,S,R)
#define G_half_sat_rte_int(TO,TI,S,R)
#define G_half_sat_rtn_int(TO,TI,S,R)
#define G_half_sat_rtp_int(TO,TI,S,R)
#define G_half_sat_rtz_int(TO,TI,S,R)
#define G_half_rte_int(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_int(TO,TI,S,R)       EXPAND(TO,TI,R,S)
#define G_half_rtp_int(TO,TI,S,R)       EXPAND(TO,TI,R,S)
#define G_half_rtz_int(TO,TI,S,R)       EXPAND(TO,TI,R,S)
#define G_half_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_uint(TO,TI,S,R)
#define G_half_sat_rte_uint(TO,TI,S,R)
#define G_half_sat_rtn_uint(TO,TI,S,R)
#define G_half_sat_rtp_uint(TO,TI,S,R)
#define G_half_sat_rtz_uint(TO,TI,S,R)
#define G_half_rte_uint(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_uint(TO,TI,S,R)      EXPAND(TO,TI,R,S)
#define G_half_rtp_uint(TO,TI,S,R)      EXPAND(TO,TI,R,S)
#define G_half_rtz_uint(TO,TI,S,R)      EXPAND(TO,TI,R,S)
#define G_half_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_long(TO,TI,S,R)
#define G_half_sat_rte_long(TO,TI,S,R)
#define G_half_sat_rtn_long(TO,TI,S,R)
#define G_half_sat_rtp_long(TO,TI,S,R)
#define G_half_sat_rtz_long(TO,TI,S,R)
#define G_half_rte_long(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_long(TO,TI,S,R)      EXPAND(TO,TI,R,S)
#define G_half_rtp_long(TO,TI,S,R)      EXPAND(TO,TI,R,S)
#define G_half_rtz_long(TO,TI,S,R)      EXPAND(TO,TI,R,S)
#define G_half_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_ulong(TO,TI,S,R)
#define G_half_sat_rte_ulong(TO,TI,S,R)
#define G_half_sat_rtn_ulong(TO,TI,S,R)
#define G_half_sat_rtp_ulong(TO,TI,S,R)
#define G_half_sat_rtz_ulong(TO,TI,S,R)
#define G_half_rte_ulong(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_ulong(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_rtp_ulong(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_rtz_ulong(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_float(TO,TI,S,R)
#define G_half_sat_rte_float(TO,TI,S,R)
#define G_half_sat_rtn_float(TO,TI,S,R)
#define G_half_sat_rtp_float(TO,TI,S,R)
#define G_half_sat_rtz_float(TO,TI,S,R)
#define G_half_rte_float(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_float(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_rtp_float(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_rtz_float(TO,TI,S,R)     EXPAND(TO,TI,R,S)
#define G_half_double(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_sat_double(TO,TI,S,R)
#define G_half_sat_rte_double(TO,TI,S,R)
#define G_half_sat_rtn_double(TO,TI,S,R)
#define G_half_sat_rtp_double(TO,TI,S,R)
#define G_half_sat_rtz_double(TO,TI,S,R)
#define G_half_rte_double(TO,TI,S,R)	CAST(TO,TI,S,R)
#define G_half_rtn_double(TO,TI,S,R)    EXPAND(TO,TI,R,S)
#define G_half_rtp_double(TO,TI,S,R)    EXPAND(TO,TI,R,S)
#define G_half_rtz_double(TO,TI,S,R)    EXPAND(TO,TI,R,S)
#define G_half_half(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_half_sat_half(TO,TI,S,R)
#define G_half_sat_rte_half(TO,TI,S,R)
#define G_half_sat_rtn_half(TO,TI,S,R)
#define G_half_sat_rtp_half(TO,TI,S,R)
#define G_half_sat_rtz_half(TO,TI,S,R)
#define G_half_rte_half(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_half_rtn_half(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_half_rtp_half(TO,TI,S,R)	NOP(TO,TI,S,R)
#define G_half_rtz_half(TO,TI,S,R)	NOP(TO,TI,S,R)

#define GEN2(TO,TI) \
    C(G_,C(TO,C(_,TI)))(TO,TI,,) \
    C(G_,C(TO,C(_sat_,TI)))(TO,TI,_sat,) \
    C(G_,C(TO,C(_sat_rte_,TI)))(TO,TI,_sat,_rte) \
    C(G_,C(TO,C(_sat_rtn_,TI)))(TO,TI,_sat,_rtn) \
    C(G_,C(TO,C(_sat_rtp_,TI)))(TO,TI,_sat,_rtp) \
    C(G_,C(TO,C(_sat_rtz_,TI)))(TO,TI,_sat,_rtz) \
    C(G_,C(TO,C(_rte_,TI)))(TO,TI,,_rte) \
    C(G_,C(TO,C(_rtn_,TI)))(TO,TI,,_rtn) \
    C(G_,C(TO,C(_rtp_,TI)))(TO,TI,,_rtp) \
    C(G_,C(TO,C(_rtz_,TI)))(TO,TI,,_rtz)

#define GEN(T) \
    GEN2(T,char) \
    GEN2(T,uchar) \
    GEN2(T,short) \
    GEN2(T,ushort) \
    GEN2(T,int) \
    GEN2(T,uint) \
    GEN2(T,long) \
    GEN2(T,ulong) \
    GEN2(T,float) \
    GEN2(T,double) \
    GEN2(T,half)

GEN(char)
GEN(uchar)
GEN(short)
GEN(ushort)
GEN(int)
GEN(uint)
GEN(long)
GEN(ulong)
GEN(float)
GEN(double)
GEN(half)

ATTR float
convert_float_rtn(int i)
{
    int s = i >> 31;
    uint u = as_uint((i + s) ^ s);
    uint lz = clz(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    uint t = u & 0xffU;
    u = (e << 23) | (u >> 8);
    return as_float((u + ((s & t) > 0)) | (s & 0x80000000));
}

ATTR float
convert_float_rtp(int i)
{
    int s = i >> 31;
    uint u = as_uint((i + s) ^ s);
    uint lz = clz(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    uint t = u & 0xffU;
    u = (e << 23) | (u >> 8);
    return as_float((u + ((~s & t) > 0)) | (s & 0x80000000));
}

ATTR float
convert_float_rtz(int i)
{
    int s = i >> 31;
    uint u = as_uint((i + s) ^ s);
    uint lz = clz(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    u = (e << 23) | (u >> 8);
    return as_float(u | (s & 0x80000000));
}

IATTR static float
cvt1f4_zu4(uint u)
{
    uint lz = clz(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    return as_float((e << 23) | (u >> 8));
}
extern AATTR("cvt1f4_zu4") float convert_float_rtn(uint);
extern AATTR("cvt1f4_zu4") float convert_float_rtz(uint);

ATTR float
convert_float_rtp(uint u)
{
    uint lz = clz(u);
    uint e = 127U + 31U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffU;
    uint t = u & 0xffU;
    u = (e << 23) | (u >> 8);
    return as_float(u + (t > 0));
}

ATTR float
convert_float_rtn(long l)
{
    long s = l >> 63;
    ulong u = as_ulong((l + s) ^ s);
    uint lz = clz(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0xffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return as_float((v + ((s & t) > 0)) | ((uint)s & 0x80000000));
}

ATTR float
convert_float_rtp(long l)
{
    long s = l >> 63;
    ulong u = as_ulong((l + s) ^ s);
    uint lz = clz(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0xffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return as_float((v + ((~s & t) > 0)) | ((uint)s & 0x80000000));
}

ATTR float
convert_float_rtz(long l)
{
    long s = l >> 63;
    ulong u = as_ulong((l + s) ^ s);
    uint lz = clz(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return as_float(v | ((uint)s & 0x80000000));
}

IATTR static float
cvt1f4_zu8(ulong u)
{
    uint lz = clz(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    return as_float((e << 23) | (uint)(u >> 40));
}
extern AATTR("cvt1f4_zu8") float convert_float_rtz(ulong);
extern AATTR("cvt1f4_zu8") float convert_float_rtn(ulong);

ATTR float
convert_float_rtp(ulong u)
{
    uint lz = clz(u);
    uint e = 127U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0xffffffffffUL;
    uint v = (e << 23) | (uint)(u >> 40);
    return as_float(v + (t > 0));
}

ATTR float
convert_float_rtn(double a)
{
    return __ocml_cvtrtn_f32_f64(a);
}

ATTR float
convert_float_rtp(double a)
{
    return __ocml_cvtrtp_f32_f64(a);
}

ATTR float
convert_float_rtz(double a)
{
    return __ocml_cvtrtz_f32_f64(a);
}

ATTR double
convert_double_rtn(long l)
{
    long s = l >> 63;
    ulong u = as_ulong((l + s) ^ s);
    uint lz = clz(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0x7ffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return as_double((u + ((s & t) > 0)) | ((ulong)s & 0x8000000000000000UL));
}

ATTR double
convert_double_rtp(long l)
{
    long s = l >> 63;
    ulong u = as_ulong((l + s) ^ s);
    uint lz = clz(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0x7ffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return as_double((u + ((~s & t) > 0)) | ((ulong)s & 0x8000000000000000UL));
}

ATTR double
convert_double_rtz(long l)
{
    long s = l >> 63;
    ulong u = as_ulong((l + s) ^ s);
    uint lz = clz(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return as_double(u | ((ulong)s & 0x8000000000000000UL));
}

IATTR static double
cvt1f8_zu8(ulong u)
{
    uint lz = clz(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    return as_double(((ulong)e << 52) | (u >> 11));
}
AATTR("cvt1f8_zu8") double convert_double_rtz(ulong);
AATTR("cvt1f8_zu8") double convert_double_rtn(ulong);

ATTR double
convert_double_rtp(ulong u)
{
    uint lz = clz(u);
    uint e = 1023U + 63U - lz;
    e = u ? e : 0;
    u = (u << lz) & 0x7fffffffffffffffUL;
    ulong t = u & 0x7ffUL;
    u = ((ulong)e << 52) | (u >> 11);
    return as_double(u + (t > 0UL));
}

ATTR half
convert_half_rtn(short s)
{
    return __ocml_cvtrtz_f16_f32((float)s);
}

ATTR half
convert_half_rtp(short s)
{
    return __ocml_cvtrtp_f16_f32((float)s);
}

ATTR half
convert_half_rtz(short s)
{
    return __ocml_cvtrtz_f16_f32((float)s);
}

IATTR static half
cvt1f2_zu2(ushort u)
{
    return __ocml_cvtrtz_f16_f32((float)u);
}
AATTR("cvt1f2_zu2") half convert_half_rtn(ushort);
AATTR("cvt1f2_zu2") half convert_half_rtz(ushort);

ATTR half
convert_half_rtp(ushort u)
{
    return __ocml_cvtrtp_f16_f32((float)u);
}

ATTR half
convert_half_rtn(int i)
{
    i = clamp(i, SHRT_MIN, SHRT_MAX);
    return __ocml_cvtrtn_f16_f32((float)i);
}

ATTR half
convert_half_rtp(int i)
{
    i = clamp(i, SHRT_MIN, SHRT_MAX);
    return __ocml_cvtrtp_f16_f32((float)i);
}

ATTR half
convert_half_rtz(int i)
{
    i = clamp(i, SHRT_MIN, SHRT_MAX);
    return __ocml_cvtrtz_f16_f32((float)i);
}

IATTR static half
cvt1f2_zu4(uint u)
{
    u = min(u, (uint)USHRT_MAX);
    return __ocml_cvtrtz_f16_f32((float)u);
}
AATTR("cvt1f2_zu4") half convert_half_rtn(uint);
AATTR("cvt1f2_zu4") half convert_half_rtz(uint);

ATTR half
convert_half_rtp(uint u)
{
    u = min(u, (uint)USHRT_MAX);
    return __ocml_cvtrtp_f16_f32((float)u);
}

ATTR half
convert_half_rtn(long l)
{
    int i = (int)clamp(l, (long)SHRT_MIN, (long)SHRT_MAX);
    return __ocml_cvtrtn_f16_f32((float)i);
}

ATTR half
convert_half_rtp(long l)
{
    int i = (int)clamp(l, (long)SHRT_MIN, (long)SHRT_MAX);
    return __ocml_cvtrtp_f16_f32((float)i);
}

ATTR half
convert_half_rtz(long l)
{
    int i = (int)clamp(l, (long)SHRT_MIN, (long)SHRT_MAX);
    return __ocml_cvtrtz_f16_f32((float)i);
}

IATTR static half
cvt1f2_zu8(ulong ul)
{
    uint u = (uint)min(ul, (ulong)USHRT_MAX);
    return __ocml_cvtrtz_f16_f32((float)u);
}
AATTR("cvt1f2_zu8") half convert_half_rtn(ulong);
AATTR("cvt1f2_zu8") half convert_half_rtz(ulong);

ATTR half
convert_half_rtp(ulong ul)
{
    uint u = (uint)min(ul, (ulong)USHRT_MAX);
    return __ocml_cvtrtp_f16_f32((float)u);
}

ATTR half
convert_half_rtp(float a)
{
    return __ocml_cvtrtp_f16_f32(a);
}

ATTR half
convert_half_rtn(float a)
{
    return __ocml_cvtrtn_f16_f32(a);
}

ATTR half
convert_half_rtz(float a)
{
    return __ocml_cvtrtz_f16_f32(a);
}

ATTR half
convert_half_rtp(double a)
{
    return __ocml_cvtrtp_f16_f64(a);
}

ATTR half
convert_half_rtn(double a)
{
    return __ocml_cvtrtn_f16_f64(a);
}

ATTR half
convert_half_rtz(double a)
{
    return __ocml_cvtrtz_f16_f64(a);
}
