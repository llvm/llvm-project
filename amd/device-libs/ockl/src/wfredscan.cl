/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"
#include "oclc.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define AS_USHORT(X) __builtin_astype(X, ushort)
#define AS_INT(X) __builtin_astype(X, int)
#define AS_UINT(X) __builtin_astype(X, uint)
#define AS_UINT2(X) __builtin_astype(X, uint2)
#define AS_LONG(X) __builtin_astype(X, long)
#define AS_ULONG(X) __builtin_astype(X, ulong)
#define AS_DOUBLE(X) __builtin_astype(X, double)
#define AS_FLOAT(X) __builtin_astype(X, float)
#define AS_HALF(X) __builtin_astype(X, half)

#define _C(X,Y) X##Y
#define C(X,Y) _C(X,Y)

// Swizzle offset macros
#define SWIZZLE_QUAD_PERM(S0,S1,S2,S3) (uint)(0x8000 | (S3 << 6) | (S2 << 4) | (S1 << 2) | S0)
#define SWIZZLE_32_LIMITED(ANDM,ORM,XORM) (uint)((XORM << 10) | (ORM << 5) | ANDM)

// DPP 9 bit control macros
#define DPP_QUAD_PERM(S0,S1,S2,S3) (uint)((S3 << 6) | (S2 << 4) | (S1 << 2) | S0)
#define DPP_ROW_SL(N) (uint)(0x100 | N)
#define DPP_ROW_SR(N) (uint)(0x110 | N)
#define DPP_ROW_RR(N) (uint)(0x120 | N)
#define DPP_WF_SL1 (uint)0x130
#define DPP_WF_RL1 (uint)0x134
#define DPP_WF_SR1 (uint)0x138
#define DPP_WF_RR1 (uint)0x13c
#define DPP_ROW_MIRROR (uint)0x140
#define DPP_ROW_HALF_MIRROR (uint)0x141
#define DPP_ROW_BCAST15 (uint)0x142
#define DPP_ROW_BCAST31 (uint)0x143

// Swizzle
#define uint_swizzle(X,Y) __llvm_amdgcn_ds_swizzle(X, Y)
#define ulong_swizzle(X,Y) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_swizzle(__x.lo, Y); \
    __r.hi = uint_swizzle(__x.hi, Y); \
    AS_ULONG(__r); \
})
#define int_swizzle(X,Y) AS_INT(uint_swizzle(AS_UINT(X),Y))
#define long_swizzle(X,Y) AS_LONG(ulong_swizzle(AS_ULONG(X),Y))
#define float_swizzle(X,Y) AS_FLOAT(uint_swizzle(AS_UINT(X),Y))
#define double_swizzle(X,Y) AS_DOUBLE(ulong_swizzle(AS_ULONG(X),Y))
#define half_swizzle(X,Y) AS_HALF((ushort)uint_swizzle((uint)AS_USHORT(X),Y))

// DPP
#define uint_dpp(X,C,R,B,F) __llvm_amdgcn_mov_dpp_i32(X,C,R,B,F)
#define ulong_dpp(X,C,R,B,F) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_dpp(__x.lo, C, R, B, F); \
    __r.hi = uint_dpp(__x.hi, C, R, B, F); \
    AS_ULONG(__r); \
})
#define int_dpp(X,C,R,B,F) AS_INT(uint_dpp(AS_UINT(X),C,R,B,F))
#define long_dpp(X,C,R,B,F) AS_LONG(ulong_dpp(AS_ULONG(X),C,R,B,F))
#define float_dpp(X,C,R,B,F) AS_FLOAT(uint_dpp(AS_UINT(X),C,R,B,F))
#define double_dpp(X,C,R,B,F) AS_DOUBLE(ulong_dpp(AS_ULONG(X),C,R,B,F))
#define half_dpp(X,C,R,B,F) AS_HALF((ushort)uint_dpp((uint)AS_USHORT(X),C,R,B,F))

// readlane
#define uint_readlane(X,L) __llvm_amdgcn_readlane(X,L)
#define ulong_readlane(X,L) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_readlane(__x.lo, L); \
    __r.hi = uint_readlane(__x.hi, L); \
    AS_ULONG(__r); \
})
#define int_readlane(X,L) AS_INT(uint_readlane(AS_UINT(X),L))
#define long_readlane(X,L) AS_LONG(ulong_readlane(AS_ULONG(X),L))
#define float_readlane(X,L) AS_FLOAT(uint_readlane(AS_UINT(X),L))
#define double_readlane(X,L) AS_DOUBLE(ulong_readlane(AS_ULONG(X),L))
#define half_readlane(X,L) AS_HALF((ushort)uint_readlane((uint)AS_USHORT(X),L))

// Select
#define uint_sel(C,B,A) ({ \
    uint __c = C; \
    (__c & B) | (~__c & A); \
})
#define ulong_sel(C,B,A) ({ \
    uint __c = C; \
    uint2 __b = AS_UINT2(B); \
    uint2 __a = AS_UINT2(A); \
    uint2 __r; \
    __r.lo = (__c & __b.lo) | (~__c & __a.lo); \
    __r.hi = (__c & __b.hi) | (~__c & __a.hi); \
    AS_ULONG(__r); \
})
#define int_sel(C,B,A) AS_INT(uint_sel(C, AS_UINT(B), AS_UINT(A)))
#define long_sel(C,B,A) AS_LONG(ulong_sel(C, AS_ULONG(B), AS_ULONG(A)))
#define float_sel(C,B,A) AS_FLOAT(uint_sel(C, AS_UINT(B), AS_UINT(A)))
#define double_sel(C,B,A) AS_DOUBLE(ulong_sel(C, AS_ULONG(B), AS_ULONG(A)))
#define half_sel(C,B,A) AS_HALF((ushort)uint_sel(C, (uint)AS_USHORT(B), (uint)AS_USHORT(A)))

#define uint_suf _u32
#define int_suf _i32
#define ulong_suf _u64
#define long_suf _i64
#define float_suf _f32
#define double_suf _f64
#define half_suf _f16

#define CATTR __attribute__((always_inline, const))
#define IATTR __attribute__((always_inline))

#define GENMIN(T) CATTR static T T##_min(T a, T b) { return a < b ? a : b; }
GENMIN(int)
GENMIN(uint)
GENMIN(long)
GENMIN(ulong)
#define float_min(A,B) __llvm_minnum_f32(A,B)
#define double_min(A,B) __llvm_minnum_f64(A,B)
#define half_min(A,B) __llvm_minnum_f16(A,B)

#define GENMAX(T) CATTR static T T##_max(T a, T b) { return a < b ? b : a; }
GENMAX(int)
GENMAX(uint)
GENMAX(long)
GENMAX(ulong)
#define float_max(A,B) __llvm_maxnum_f32(A,B)
#define double_max(A,B) __llvm_maxnum_f64(A,B)
#define half_max(A,B) __llvm_maxnum_f16(A,B)

#define ADD(X,Y) (X + Y)
#define uint_add(X,Y) ADD(X,Y)
#define int_add(X,Y) ADD(X,Y)
#define ulong_add(X,Y) ADD(X,Y)
#define long_add(X,Y) ADD(X,Y)
#define float_add(X,Y) ADD(X,Y)
#define double_add(X,Y) ADD(X,Y)
#define half_add(X,Y) ADD(X,Y)

#define OR(X,Y) (X | Y)
#define uint_or(X,Y) OR(X,Y)
#define int_or(X,Y) OR(X,Y)
#define ulong_or(X,Y) OR(X,Y)
#define long_or(X,Y) OR(X,Y)

#define AND(X,Y) (X | Y)
#define uint_and(X,Y) AND(X,Y)
#define int_and(X,Y) AND(X,Y)
#define ulong_and(X,Y) AND(X,Y)
#define long_and(X,Y) AND(X,Y)

#define XOR(X,Y) (X ^ Y)
#define uint_xor(X,Y) XOR(X,Y)
#define int_xor(X,Y) XOR(X,Y)
#define ulong_xor(X,Y) XOR(X,Y)
#define long_xor(X,Y) XOR(X,Y)


// Reduce with operation OP over full wave using swizzle
// Input in x, r is result
#define RED_SWIZZLE_FULL(T,OP) \
    T v; \
 \
    v = T##_swizzle(x, SWIZZLE_QUAD_PERM(0x1,0x0,0x3,0x2)); \
    r = T##_##OP(x, v); \
 \
    v = T##_swizzle(r, SWIZZLE_QUAD_PERM(0x2,0x3,0x0,0x1)); \
    r = T##_##OP(r, v); \
 \
    v = T##_swizzle(r, SWIZZLE_32_LIMITED(0x1f,0x00,0x04)); \
    r = T##_##OP(r, v); \
 \
    v = T##_swizzle(r, SWIZZLE_32_LIMITED(0x1f,0x00,0x08)); \
    r = T##_##OP(r, v); \
 \
    v = T##_swizzle(r, SWIZZLE_32_LIMITED(0x1f,0x00,0x10)); \
    r = T##_##OP(r, v); \
 \
    r = T##_##OP(T##_readlane(r,0), T##_readlane(r,32))

// Reduce with operation OP over partial wave using swizzle
// Input in x, r is result
#define RED_SWIZZLE_PART(T,OP,ID) \
    uint e; \
    T v, t; \
 \
    t = T##_swizzle(x,    SWIZZLE_QUAD_PERM(0x1,0x0,0x3,0x2)); \
    e = uint_swizzle(~0u, SWIZZLE_QUAD_PERM(0x1,0x0,0x3,0x2)); \
    v = T##_sel(e, t, ID); \
    r = T##_##OP(x, v); \
 \
    t = T##_swizzle(r,    SWIZZLE_QUAD_PERM(0x2,0x3,0x0,0x1)); \
    e = uint_swizzle(~0u, SWIZZLE_QUAD_PERM(0x2,0x3,0x0,0x1)); \
    v = T##_sel(e, t, ID); \
    r = T##_##OP(r, v); \
 \
    t = T##_swizzle(r,    SWIZZLE_32_LIMITED(0x1f,0x00,0x04)); \
    e = uint_swizzle(~0u, SWIZZLE_32_LIMITED(0x1f,0x00,0x04)); \
    v = T##_sel(e, t, ID); \
    r = T##_##OP(r, v); \
 \
    t = T##_swizzle(r,    SWIZZLE_32_LIMITED(0x1f,0x00,0x08)); \
    e = uint_swizzle(~0u, SWIZZLE_32_LIMITED(0x1f,0x00,0x08)); \
    v = T##_sel(e, t, ID); \
    r = T##_##OP(r, v); \
 \
    t = T##_swizzle(r,    SWIZZLE_32_LIMITED(0x1f,0x00,0x10)); \
    e = uint_swizzle(~0u, SWIZZLE_32_LIMITED(0x1f,0x00,0x10)); \
    v = T##_sel(e, t, ID); \
    r = T##_##OP(r, v); \
 \
    t = T##_readlane(r, 32); \
    v = (__llvm_amdgcn_read_exec_hi() & 1) ? t : ID; \
    r = T##_##OP(T##_readlane(r, 0), v)


// Reduce with operation OP over full wave using DPP
// Input in x, r is result
#define RED_DPP_FULL(T,OP) \
    T v; \
 \
    v = T##_dpp(x, DPP_QUAD_PERM(0x1,0x0,0x3,0x2), 0xf, 0xf, true); \
    r = T##_##OP(x, v); \
 \
    v = T##_dpp(r, DPP_QUAD_PERM(0x2,0x3,0x0,0x1), 0xf, 0xf, true); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(r, DPP_ROW_SR(4), 0xf, 0xa, true); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(r, DPP_ROW_SR(8), 0xf, 0x8, true); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(r, DPP_ROW_BCAST15, 0xe, 0x8, true); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(r, DPP_ROW_BCAST31, 0x8, 0x8, true); \
    r = T##_##OP(r, v); \
 \
    r = T##_readlane(r, 63)

// Reduce with operation OP over partial wave using DPP
// Input in x, r is result
#define RED_DPP_PART(T,OP,ID) \
    if (ID == (T)0) { \
        T v; \
 \
        v = T##_dpp(x, DPP_QUAD_PERM(0x1,0x0,0x3,0x2), 0xf, 0xf, true); \
        r = T##_##OP(x, v); \
 \
        v = T##_dpp(r, DPP_QUAD_PERM(0x2,0x3,0x0,0x1), 0xf, 0xf, true); \
        r = T##_##OP(r, v); \
 \
        v = T##_dpp(r, DPP_ROW_SL(4), 0xf, 0x5, true); \
        r = T##_##OP(r, v); \
 \
        v = T##_dpp(r, DPP_ROW_SL(8), 0xf, 0x1, true); \
        r = T##_##OP(r, v); \
 \
        v = T##_dpp(r, DPP_WF_SL1, 0xf, 0x8, true); \
        v = T##_dpp(v, DPP_ROW_MIRROR, 0xf, 0x1, true); \
        r = T##_##OP(r, v); \
    } else { \
        T t, v; \
        uint e; \
 \
        t = T##_dpp(x,    DPP_QUAD_PERM(0x1,0x0,0x3,0x2), 0xf, 0xf, true); \
        e = uint_dpp(~0u, DPP_QUAD_PERM(0x1,0x0,0x3,0x2), 0xf, 0xf, true); \
        v = T##_sel(e, t, ID); \
        r = T##_##OP(x, v); \
 \
        t = T##_dpp(r,    DPP_QUAD_PERM(0x2,0x3,0x0,0x1), 0xf, 0xf, true); \
        e = uint_dpp(~0u, DPP_QUAD_PERM(0x2,0x3,0x0,0x1), 0xf, 0xf, true); \
        v = T##_sel(e, t, ID); \
        r = T##_##OP(r, v); \
 \
        t = T##_dpp(r,    DPP_ROW_SL(4), 0xf, 0x5, true); \
        e = uint_dpp(~0u, DPP_ROW_SL(4), 0xf, 0x5, true); \
        v = T##_sel(e, t, ID); \
        r = T##_##OP(r, v); \
 \
        t = T##_dpp(r,    DPP_ROW_SL(8), 0xf, 0x1, true); \
        e = uint_dpp(~0u, DPP_ROW_SL(8), 0xf, 0x1, true); \
        v = T##_sel(e, t, ID); \
        r = T##_##OP(r, v); \
 \
        t = T##_dpp(r,    DPP_WF_SL1, 0xf, 0x8, true); \
        e = uint_dpp(~0u, DPP_WF_SL1, 0xf, 0x8, true); \
        t = T##_dpp(t,  DPP_ROW_MIRROR, 0xf, 0x1, true); \
        e = uint_dpp(e, DPP_ROW_MIRROR, 0xf, 0x1, true); \
        v = T##_sel(e, t, ID); \
        r = T##_##OP(r, v); \
    } \
 \
    T t32 = T##_readlane(r, 32); \
    T v32 = (__llvm_amdgcn_read_exec_hi() & 1) ? t32 : ID; \
    r = T##_##OP(T##_readlane(r, 0), v32)

// Inclusive scan with operation OP using swizzle
// Input is x, l is lane, output is s
#define ISCAN_SWIZZLE(T,OP,ID) \
    T v; \
 \
    v = T##_swizzle(x, SWIZZLE_32_LIMITED(0x1e,0x00,0x00)); \
    v = (l & 1) ? v : ID; \
    s = T##_##OP(x, v); \
 \
    v = T##_swizzle(s, SWIZZLE_32_LIMITED(0x1c,0x01,0x00)); \
    v = (l & 2) ? v : ID; \
    s = T##_##OP(s, v); \
 \
    v = T##_swizzle(s, SWIZZLE_32_LIMITED(0x18,0x03,0x00)); \
    v = (l & 4) ? v : ID; \
    s = T##_##OP(s, v); \
 \
    v = T##_swizzle(s, SWIZZLE_32_LIMITED(0x10,0x07,0x00)); \
    v = (l & 8) ? v : ID; \
    s = T##_##OP(s, v); \
 \
    v = T##_swizzle(s, SWIZZLE_32_LIMITED(0x00,0x0f,0x00)); \
    v = (l & 16) ? v : ID; \
    s = T##_##OP(s, v); \
 \
    v = T##_readlane(s, 31); \
    v = (l & 32) ? v : ID; \
    s = T##_##OP(s, v)


// Inclusive scan with operation OP using DPP
// Input is x, l is lane, output is s
#define ISCAN_DPP(T,OP,ID) \
    if (ID == (T)0) { \
        T v; \
 \
        v = T##_dpp(x, DPP_ROW_SR(1), 0xf, 0xf, true); \
        s = T##_##OP(x, v); \
 \
        v = T##_dpp(s, DPP_ROW_SR(2), 0xf, 0xf, true); \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_SR(4), 0xf, 0xf, true); \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_SR(8), 0xf, 0xf, true); \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_BCAST15, 0xf, 0xf, true); \
        v = (l & 0x10) ? v : ID; \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_BCAST31, 0xf, 0xf, true); \
        s = T##_##OP(s, v); \
    } else { \
        T v; \
 \
        v = T##_dpp(x, DPP_ROW_SR(1), 0xf, 0xf, true); \
        v = l >= 1 ? v : ID; \
        s = T##_##OP(x, v); \
 \
        v = T##_dpp(s, DPP_ROW_SR(2), 0xf, 0xf, true); \
        v = l >= 2 ? v : ID; \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_SR(4), 0xf, 0xf, true); \
        v = l >= 4 ? v : ID; \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_SR(8), 0xf, 0xf, true); \
        v = l >= 8 ? v : ID; \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_BCAST15, 0xf, 0xf, true); \
        v = (l & 0x10) ? v : ID; \
        s = T##_##OP(s, v); \
 \
        v = T##_dpp(s, DPP_ROW_BCAST31, 0xf, 0xf, true); \
        v = (l & 0x20) ? v : ID; \
        s = T##_##OP(s, v); \
    }

// Shift right 1 on entire wavefront using swizzle
// input is s, l is lane, output is s
#define SR1_SWIZZLE(T,ID) \
    T v; \
    T t = s; \
 \
    s = T##_swizzle(t, SWIZZLE_QUAD_PERM(0x0,0x0,0x1,0x2)); \
 \
    v = T##_swizzle(t, SWIZZLE_32_LIMITED(0x18, 0x03, 0x00)); \
    s = (l & 0x7) == 0x4 ? v : s; \
 \
    v = T##_swizzle(t, SWIZZLE_32_LIMITED(0x10, 0x07, 0x00)); \
    s = (l & 0xf) == 0x8 ? v : s; \
 \
    v = T##_swizzle(t, SWIZZLE_32_LIMITED(0x00, 0x0f, 0x00)); \
    s = (l & 0x1f) == 0x10 ? v : s; \
 \
    v = T##_readlane(t, 31); \
    s = l == 32 ? v : s; \
 \
    s = l == 0 ? ID : s

// Shift right 1 on entire wavefront using DPP
// input is s, l is lane, output is s
#define SR1_DPP(T,ID) \
    s = T##_dpp(s, DPP_WF_SR1, 0xf, 0xf, true); \
    if (ID != (T)0) {\
        s = l == 0 ? ID : s; \
    }

IATTR static bool
fullwave(void)
{
    return __llvm_ctpop_i32(__llvm_amdgcn_read_exec_lo()) +
           __llvm_ctpop_i32(__llvm_amdgcn_read_exec_hi()) == 64;
}

#define GENRED(T,OP,ID) \
IATTR T \
C(__ockl_wfred_,C(OP,T##_suf))(T x) \
{ \
    T r; \
    if (fullwave()) { \
        if (__oclc_ISA_version() < 800) { \
            RED_SWIZZLE_FULL(T,OP); \
        } else { \
            RED_DPP_FULL(T,OP); \
        } \
    } else { \
        if (__oclc_ISA_version() < 800) { \
            RED_SWIZZLE_PART(T,OP,ID); \
        } else { \
            RED_DPP_PART(T,OP,ID); \
        } \
    } \
    return r; \
}

#define GENSCAN(T,OP,ID) \
IATTR T \
C(__ockl_wfscan_,C(OP,T##_suf))(T x, bool inclusive) \
{ \
    T s; \
    uint l = __ockl_activelane_u32(); \
 \
    if (__oclc_ISA_version() < 800) { \
        ISCAN_SWIZZLE(T,OP,ID); \
    } else { \
        ISCAN_DPP(T,OP,ID); \
    } \
 \
    if (!inclusive) { \
        if (__oclc_ISA_version() < 800) { \
            SR1_SWIZZLE(T,ID); \
        } else { \
            SR1_DPP(T,ID); \
        } \
    } \
 \
    return s; \
}

#define GEN(T,OP,ID) \
    GENRED(T,OP,ID) \
    GENSCAN(T,OP,ID)

GEN(int,add,0)
GEN(uint,add,0u)
GEN(long,add,0L)
GEN(ulong,add,0UL)
GEN(float,add,0.0f)
GEN(double,add,0.0)
GEN(half,add,0.0h)

GEN(int,min,INT_MAX)
GEN(uint,min,UINT_MAX)
GEN(long,min,LONG_MAX)
GEN(ulong,min,ULONG_MAX)
GEN(float,min,INFINITY)
GEN(double,min,(double)INFINITY)
GEN(half,min,(half)INFINITY)

GEN(int,max,INT_MIN)
GEN(uint,max,0u)
GEN(long,max,LONG_MIN)
GEN(ulong,max,0UL)
GEN(float,max,-INFINITY)
GEN(double,max,-(double)INFINITY)
GEN(half,max,-(half)INFINITY)

GEN(int,and,~0)
GEN(uint,and,~0u)
GEN(long,and,~0L)
GEN(ulong,and,~0UL)

GEN(int,or,0)
GEN(uint,or,0u)
GEN(long,or,0L)
GEN(ulong,or,0UL)

GEN(int,xor,0)
GEN(uint,xor,0u)
GEN(long,xor,0L)
GEN(ulong,xor,0UL)

