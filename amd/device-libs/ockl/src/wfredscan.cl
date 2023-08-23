/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

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
#define DPP_ROW_SHARE(N) (uint)(0x150 | N)
#define DPP_ROW_XMASK(N) (uint)(0x160 | N)

// Swizzle
#define uint_swizzle(X,Y) __builtin_amdgcn_ds_swizzle(X, Y)
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

// DPP16
#define uint_dpp(ID,X,C,R,B,W) __builtin_amdgcn_update_dpp(ID,X,C,R,B,W)
#define ulong_dpp(ID,X,C,R,B,W) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_dpp((uint)ID, __x.lo, C, R, B, W); \
    __r.hi = uint_dpp((uint)(ID >> 32), __x.hi, C, R, B, W); \
    AS_ULONG(__r); \
})
#define int_dpp(ID,X,C,R,B,W) AS_INT(uint_dpp(AS_UINT(ID),AS_UINT(X),C,R,B,W))
#define long_dpp(ID,X,C,R,B,W) AS_LONG(ulong_dpp(AS_ULONG(ID),AS_ULONG(X),C,R,B,W))
#define float_dpp(ID,X,C,R,B,W) AS_FLOAT(uint_dpp(AS_UINT(ID),AS_UINT(X),C,R,B,W))
#define double_dpp(ID,X,C,R,B,W) AS_DOUBLE(ulong_dpp(AS_ULONG(ID),AS_ULONG(X),C,R,B,W))
#define half_dpp(ID,X,C,R,B,W) AS_HALF((ushort)uint_dpp((uint)AS_USHORT(ID),(uint)AS_USHORT(X),C,R,B,W))

// DPP8
#define uint_dpp8(X,S) __builtin_amdgcn_mov_dpp8(X,S)
#define ulong_dpp8(X,S) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_dpp8(__x.lo, S); \
    __r.hi = uint_dpp8(__x.hi, S); \
    AS_ULONG(__r); \
})
#define int_dpp8(X,S) AS_INT(uint_dpp8(AS_UINT(X),S))
#define long_dpp8(X,S) AS_LONG(ulong_dpp8(AS_ULONG(X),S))
#define float_dpp8(X,S) AS_FLOAT(uint_dpp8(AS_UINT(X),S))
#define double_dpp8(X,S) AS_DOUBLE(ulong_dpp8(AS_ULONG(X),S))
#define half_dpp8(X,S) AS_HALF((ushort)uint_dpp8((uint)AS_USHORT(X),S))

// permlane16
#define uint_permlane16(ID,X,S0,S1,W) __builtin_amdgcn_permlane16(ID,X,S0,S1,false,W)
#define ulong_permlane16(ID,X,S0,S1,W) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_permlane16((uint)ID,__x.lo,S0,S1,W); \
    __r.hi = uint_permlane16((uint)(ID>>32),__x.hi,S0,S1,W); \
    AS_ULONG(__r); \
})
#define int_permlane16(ID,X,S0,S1,W) AS_INT(uint_permlane16(AS_UINT(ID),AS_UINT(X),S0,S1,W))
#define long_permlane16(ID,X,S0,S1,W) AS_LONG(ulong_permlane16(AS_ULONG(ID),AS_ULONG(X),S0,S1,W))
#define float_permlane16(ID, X,S0,S1,W) AS_FLOAT(uint_permlane16(AS_UINT(ID),AS_UINT(X),S0,S1,W))
#define double_permlane16(ID, X,S0,S1,W) AS_DOUBLE(ulong_permlane16(AS_ULONG(ID),AS_ULONG(X),S0,S1,W))
#define half_permlane16(ID,X,S0,S1,W) AS_HALF((ushort)uint_permlane16((uint)AS_USHORT(ID),(uint)AS_USHORT(X),S0,S1,W))

// permlanex16
#define uint_permlanex16(ID,X,S0,S1,W) __builtin_amdgcn_permlanex16(ID,X,S0,S1,false,W)
#define ulong_permlanex16(ID,X,S0,S1,W) ({ \
    uint2 __x = AS_UINT2(X); \
    uint2 __r; \
    __r.lo = uint_permlanex16((uint)ID,__x.lo,S0,S1,W); \
    __r.hi = uint_permlanex16((uint)(ID>>32),__x.hi,S0,S1,W); \
    AS_ULONG(__r); \
})
#define int_permlanex16(ID,X,S0,S1,W) AS_INT(uint_permlanex16(AS_UINT(ID),AS_UINT(X),S0,S1,W))
#define long_permlanex16(ID,X,S0,S1,W) AS_LONG(ulong_permlanex16(AS_ULONG(ID),AS_ULONG(X),S0,S1,W))
#define float_permlanex16(ID, X,S0,S1,W) AS_FLOAT(uint_permlanex16(AS_UINT(ID),AS_UINT(X),S0,S1,W))
#define double_permlanex16(ID, X,S0,S1,W) AS_DOUBLE(ulong_permlanex16(AS_ULONG(ID),AS_ULONG(X),S0,S1,W))
#define half_permlanex16(ID,X,S0,S1,W) AS_HALF((ushort)uint_permlanex16((uint)AS_USHORT(ID),(uint)AS_USHORT(X),S0,S1,W))

// readlane
#define uint_readlane(X,L) __builtin_amdgcn_readlane(X,L)
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

#define CATTR __attribute__((const))
#define IATTR

#define GENMIN(T) CATTR static T T##_min(T a, T b) { return a < b ? a : b; }
GENMIN(int)
GENMIN(uint)
GENMIN(long)
GENMIN(ulong)
#define float_min(A,B) __builtin_fminf(A,B)
#define double_min(A,B) __builtin_fmin(A,B)
#define half_min(A,B) __builtin_fminf16(A,B)

#define GENMAX(T) CATTR static T T##_max(T a, T b) { return a < b ? b : a; }
GENMAX(int)
GENMAX(uint)
GENMAX(long)
GENMAX(ulong)
#define float_max(A,B) __builtin_fmaxf(A,B)
#define double_max(A,B) __builtin_fmax(A,B)
#define half_max(A,B) __builtin_fmaxf16(A,B)

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

#define AND(X,Y) (X & Y)
#define uint_and(X,Y) AND(X,Y)
#define int_and(X,Y) AND(X,Y)
#define ulong_and(X,Y) AND(X,Y)
#define long_and(X,Y) AND(X,Y)

#define XOR(X,Y) (X ^ Y)
#define uint_xor(X,Y) XOR(X,Y)
#define int_xor(X,Y) XOR(X,Y)
#define ulong_xor(X,Y) XOR(X,Y)
#define long_xor(X,Y) XOR(X,Y)


#define GENRED7_FULL(T,OP,ID,IDZ) \
static T \
red7_full_##T##_##OP(T x) \
{ \
    T v, r; \
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
    r = T##_##OP(T##_readlane(r,0), T##_readlane(r,32)); \
 \
    return r; \
}

#define GENRED7_PART(T,OP,ID,IDZ) \
static T \
red7_part_##T##_##OP(T x) \
{ \
    T r; \
    if (IDZ) { \
        T v; \
 \
        v = T##_swizzle(x,    SWIZZLE_QUAD_PERM(0x1,0x0,0x3,0x2)); \
        r = T##_##OP(x, v); \
 \
        v = T##_swizzle(r,    SWIZZLE_QUAD_PERM(0x2,0x3,0x0,0x1)); \
        r = T##_##OP(r, v); \
 \
        v = T##_swizzle(r,    SWIZZLE_32_LIMITED(0x1f,0x00,0x04)); \
        r = T##_##OP(r, v); \
 \
        v = T##_swizzle(r,    SWIZZLE_32_LIMITED(0x1f,0x00,0x08)); \
        r = T##_##OP(r, v); \
 \
        v = T##_swizzle(r,    SWIZZLE_32_LIMITED(0x1f,0x00,0x10)); \
        r = T##_##OP(r, v); \
 \
        v = T##_readlane(r, 32); \
        v = (__builtin_amdgcn_read_exec_hi() & 1) ? v : ID; \
        r = T##_##OP(T##_readlane(r, 0), v); \
    } else { \
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
        v = (__builtin_amdgcn_read_exec_hi() & 1) ? t : ID; \
        r = T##_##OP(T##_readlane(r, 0), v); \
    } \
 \
    return r; \
}

#define GENRED7(T,OP,ID,IDZ) \
    GENRED7_FULL(T,OP,ID,IDZ) \
    GENRED7_PART(T,OP,ID,IDZ)

#define GENRED89(T,OP,ID,IDZ) \
__attribute__((target("dpp"))) static T \
red89_##T##_##OP(T x) \
{ \
    T r, v; \
 \
    v = T##_dpp(ID, x, DPP_ROW_SL(1), 0xf, 0xf, IDZ); \
    r = T##_##OP(x, v); \
 \
    v = T##_dpp(ID, r, DPP_ROW_SL(2), 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(ID, r, DPP_ROW_SL(4), 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(ID, r, DPP_ROW_SL(8), 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(ID, r, DPP_WF_SL1, 0xf, 0xf, IDZ); \
    v = T##_dpp(ID, v, DPP_ROW_MIRROR, 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    v = T##_readlane(r, 32); \
    v = (__builtin_amdgcn_read_exec_hi() & 1) ? v : ID; \
    r = T##_##OP(T##_readlane(r, 0), v); \
 \
    return r; \
}

#define GENRED10(T,OP,ID,IDZ) \
__attribute__((target("dpp,gfx10-insts"))) static T \
red10_##T##_##OP(T x) \
{ \
    T r, v; \
 \
    v = T##_dpp(ID, x, DPP_ROW_SL(1), 0xf, 0xf, IDZ); \
    r = T##_##OP(x, v); \
 \
    v = T##_dpp(ID, r, DPP_ROW_SL(2), 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(ID, r, DPP_ROW_SL(4), 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    v = T##_dpp(ID, r, DPP_ROW_SL(8), 0xf, 0xf, IDZ); \
    r = T##_##OP(r, v); \
 \
    r = T##_dpp(ID, r, DPP_ROW_SHARE(0), 0xf, 0xf, IDZ); \
 \
    v = T##_permlanex16(ID, r, 0, 0, IDZ); \
    r = T##_##OP(r, v); \
 \
    if (__oclc_wavefrontsize64) { \
        T v = T##_readlane(r, 32); \
        v = (__builtin_amdgcn_read_exec_hi() & 1) ? v : ID; \
        r =  T##_##OP(T##_readlane(r, 0), v); \
    } \
 \
    return r; \
}

#define GENISCAN7(T,OP,ID,IDZ) \
static T \
iscan7_##T##_##OP(T x, uint l) \
{ \
    T s, v; \
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
    v = l > 31 ? v : ID; \
    s = T##_##OP(s, v); \
 \
    return s; \
}

#define GENISCAN89(T,OP,ID,IDZ) \
__attribute__((target("dpp"))) static T \
iscan89_##T##_##OP(T x, uint l) \
{ \
    T s, v; \
 \
    v = T##_dpp(ID, x, DPP_ROW_SR(1), 0xf, 0xf, IDZ); \
    s = T##_##OP(x, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_SR(2), 0xf, 0xf, IDZ); \
    s = T##_##OP(s, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_SR(4), 0xf, 0xf, IDZ); \
    s = T##_##OP(s, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_SR(8), 0xf, 0xf, IDZ); \
    s = T##_##OP(s, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_BCAST15, 0xa, 0xf, false); \
    s = T##_##OP(s, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_BCAST31, 0xc, 0xf, false); \
    s = T##_##OP(s, v); \
 \
    return s; \
}

#define GENISCAN10(T,OP,ID,IDZ) \
__attribute__((target("dpp,gfx10-insts"))) static T \
iscan10_##T##_##OP(T x, uint l) \
{ \
    T s, v; \
 \
    v = T##_dpp(ID, x, DPP_ROW_SR(1), 0xf, 0xf, IDZ); \
    s = T##_##OP(x, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_SR(2), 0xf, 0xf, IDZ); \
    s = T##_##OP(s, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_SR(4), 0xf, 0xf, IDZ); \
    s = T##_##OP(s, v); \
 \
    v = T##_dpp(ID, s, DPP_ROW_SR(8), 0xf, 0xf, IDZ); \
    s = T##_##OP(s, v); \
 \
    v = T##_permlanex16(ID, s, 0xffffffff, 0xffffffff, IDZ); \
    v = (l & 0x10) ? v : ID; \
    s = T##_##OP(s, v); \
 \
    if (__oclc_wavefrontsize64) { \
        v = T##_readlane(s, 31); \
        v = l > 31 ? v : ID; \
        s = T##_##OP(s, v); \
    } \
 \
     return s; \
}

#define GENSR1_7(T,OP,ID,IDZ) \
static T \
sr1_7_##T##_##OP(T s, uint l) \
{ \
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
    s = l == 0 ? ID : s; \
 \
    return s; \
}


#define GENSR1_89(T,OP,ID,IDZ) \
__attribute__((target("dpp"))) static T \
sr1_89_##T##_##OP(T s, uint l) \
{ \
    return T##_dpp(ID, s, DPP_WF_SR1, 0xf, 0xf, IDZ); \
}

#define GENSR1_10(T,OP,ID,IDZ) \
__attribute((target("dpp,gfx10-insts"))) static T \
sr1_10_##T##_##OP(T s, uint l) \
{ \
    T t = T##_dpp(ID, s, DPP_ROW_SR(1), 0xf, 0xf, IDZ); \
    T v = T##_permlanex16(ID, s, 0xffffffff, 0xffffffff, IDZ); \
    if (__oclc_wavefrontsize64) { \
        T w = T##_readlane(s, 31); \
        v = l == 32 ? w : v; \
        s = ((l == 32) | ((l & 0x1f) == 0x10)) ? v : t; \
    } else {\
        s = l == 16 ? v : t; \
    } \
 \
    return s; \
}

IATTR static bool
fullwave(void)
{
    if (__oclc_wavefrontsize64) {
        return __builtin_popcountl(__builtin_amdgcn_read_exec()) == 64;
    } else {
        return __builtin_popcount(__builtin_amdgcn_read_exec_lo()) == 32;
    }
}

#define GENRED(T,OP,ID,IDZ) \
GENRED7(T,OP,ID,IDZ) \
GENRED89(T,OP,ID,IDZ) \
GENRED10(T,OP,ID,IDZ) \
IATTR T \
C(__ockl_wfred_,C(OP,T##_suf))(T x) \
{ \
    T r; \
    if (__oclc_ISA_version < 8000) { \
         if (fullwave()) { \
             r = red7_full_##T##_##OP(x); \
         } else { \
             r = red7_part_##T##_##OP(x); \
         } \
    } else if (__oclc_ISA_version < 10000) { \
        r = red89_##T##_##OP(x); \
    } else { \
        r = red10_##T##_##OP(x); \
    } \
    return r; \
}

#define GENSCAN(T,OP,ID,IDZ) \
GENISCAN7(T,OP,ID,IDZ) \
GENISCAN89(T,OP,ID,IDZ) \
GENISCAN10(T,OP,ID,IDZ) \
GENSR1_7(T,OP,ID,IDZ) \
GENSR1_89(T,OP,ID,IDZ) \
GENSR1_10(T,OP,ID,IDZ) \
IATTR T \
C(__ockl_wfscan_,C(OP,T##_suf))(T x, bool inclusive) \
{ \
    T s; \
    uint l = __ockl_lane_u32(); \
 \
    if (__oclc_ISA_version < 8000) { \
        s = iscan7_##T##_##OP(x, l); \
    } else  if (__oclc_ISA_version < 10000)  { \
        s = iscan89_##T##_##OP(x, l); \
    } else { \
        s = iscan10_##T##_##OP(x, l); \
    } \
 \
    if (!inclusive) { \
        if (__oclc_ISA_version < 8000) { \
            s = sr1_7_##T##_##OP(s, l); \
        } else  if (__oclc_ISA_version < 10000)  { \
            s = sr1_89_##T##_##OP(s, l); \
        } else { \
            s = sr1_10_##T##_##OP(s, l); \
        } \
    } \
 \
    return s; \
}

#define GEN(T,OP,ID,IDZ) \
    GENRED(T,OP,ID,IDZ) \
    GENSCAN(T,OP,ID,IDZ)

GEN(int,add,0,1)
GEN(uint,add,0u,1)
GEN(long,add,0L,1)
GEN(ulong,add,0UL,1)
GEN(float,add,0.0f,1)
GEN(double,add,0.0,1)
GEN(half,add,0.0h,1)

GEN(int,min,INT_MAX,0)
GEN(uint,min,UINT_MAX,0)
GEN(long,min,LONG_MAX,0)
GEN(ulong,min,ULONG_MAX,0)
GEN(float,min,INFINITY,0)
GEN(double,min,(double)INFINITY,0)
GEN(half,min,(half)INFINITY,0)

GEN(int,max,INT_MIN,0)
GEN(uint,max,0u,1)
GEN(long,max,LONG_MIN,0)
GEN(ulong,max,0UL,1)
GEN(float,max,-INFINITY,0)
GEN(double,max,-(double)INFINITY,0)
GEN(half,max,-(half)INFINITY,0)

GEN(int,and,~0,0)
GEN(uint,and,~0u,0)
GEN(long,and,~0L,0)
GEN(ulong,and,~0UL,0)

GEN(int,or,0,1)
GEN(uint,or,0u,1)
GEN(long,or,0L,1)
GEN(ulong,or,0UL,1)

GEN(int,xor,0,1)
GEN(uint,xor,0u,1)
GEN(long,xor,0L,1)
GEN(ulong,xor,0UL,1)

