/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "ockl.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define CATTR __attribute__((always_inline, const))
#define AS_UCHAR4(X) __builtin_astype(X, uchar4)

CATTR uint
OCKL_MANGLE_U32(bfm)(uint w, uint s)
{
    // TODO check that this results in v_bfm_b32
    return ((1U << w) - 1U) << s;
}

CATTR int
OCKL_MANGLE_I32(bfe)(int a, uint s, uint w)
{
    return __llvm_amdgcn_sbfe_i32(a, s, w);
}

CATTR uint
OCKL_MANGLE_U32(bfe)(uint a, uint s, uint w)
{
    return __llvm_amdgcn_ubfe_i32(a, s, w);
}

CATTR uint
OCKL_MANGLE_U32(bitalign)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_alignbit(a, b, c);
}

CATTR uint
OCKL_MANGLE_U32(bytealign)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_alignbyte(a, b, c);
}

CATTR uint
OCKL_MANGLE_U32(lerp)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_lerp(a, b, c);
}

CATTR float
OCKL_MANGLE_F32(max3)(float a, float b, float c)
{
    return __llvm_maxnum_f32(__llvm_maxnum_f32(a, b), c);
}

CATTR float
OCKL_MANGLE_F32(median3)(float a, float b, float c)
{
    return __llvm_amdgcn_fmed3_f32(a, b, c);
}

CATTR float
OCKL_MANGLE_F32(min3)(float a, float b, float c)
{
    return __llvm_minnum_f32(__llvm_minnum_f32(a, b), c);
}

CATTR half
OCKL_MANGLE_F16(max3)(half a, half b, half c)
{
    return __llvm_maxnum_f16(__llvm_maxnum_f16(a, b), c);
}

CATTR half
OCKL_MANGLE_F16(median3)(half a, half b, half c)
{
    return __llvm_amdgcn_fmed3_f16(a, b, c);
}

CATTR half
OCKL_MANGLE_F16(min3)(half a, half b, half c)
{
    return __llvm_minnum_f16(__llvm_minnum_f16(a, b), c);
}

CATTR int
OCKL_MANGLE_I32(max3)(int a, int b, int c)
{
    // TODO check that this results in v_max3_i32
    int a1 = a > b ? a : b;
    return a1 > c ? a1 : c;
}

CATTR int
OCKL_MANGLE_I32(median3)(int a, int b, int c)
{
    // TODO check that this results in v_med3_i32
    int a1 = a < b ? a : b;
    int b1 = a > b ? a : b;
    int c1 = a1 > c ? a1 : c;
    return b1 < c1 ? b1 : c1;
}

CATTR int
OCKL_MANGLE_I32(min3)(int a, int b, int c)
{
    // TODO check that this results in v_min3_i32
    int a1 = a < b ? a : b;
    return a1 < c ? a1 : c;
}

CATTR uint
OCKL_MANGLE_U32(max3)(uint a, uint b, uint c)
{
    // TODO check that this results in v_max3_u32
    uint a1 = a > b ? a : b;
    return a1 > c ? a1 : c;
}

CATTR uint
OCKL_MANGLE_U32(median3)(uint a, uint b, uint c)
{
    // TODO check that this results in v_med3_u32
    uint a1 = a < b ? a : b;
    uint b1 = a > b ? a : b;
    uint c1 = a1 > c ? a1 : c;
    return b1 < c1 ? b1 : c1;
}

CATTR uint
OCKL_MANGLE_U32(min3)(uint a, uint b, uint c)
{
    // TODO check that this results in v_min3_u32
    uint a1 = a < b ? a : b;
    return a1 < c ? a1 : c;
}

CATTR uint
OCKL_MANGLE_U32(msad)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_msad_u8(a, b, c);
}

CATTR ulong
OCKL_MANGLE_U64(mqsad)(ulong a, uint b, ulong c)
{
    return __llvm_amdgcn_mqsad_pk_u16_u8(a, b, c);
}

CATTR uint
OCKL_MANGLE_U32(pack)(float4 a)
{
    return __llvm_amdgcn_cvt_pk_u8_f32(a.s3, 3,
             __llvm_amdgcn_cvt_pk_u8_f32(a.s2, 2,
               __llvm_amdgcn_cvt_pk_u8_f32(a.s1, 1,
                 __llvm_amdgcn_cvt_pk_u8_f32(a.s0, 0, 0))));
}

CATTR ulong
OCKL_MANGLE_U64(qsad)(ulong a, uint b, ulong c)
{
    return __llvm_amdgcn_qsad_pk_u16_u8(a, b, c);
}

CATTR uint
OCKL_MANGLE_U32(sad)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_sad_u8(a, b, c);
}

CATTR uint
OCKL_MANGLE_U32(sadd)(uint a, uint b, uint c)
{
    // TODO check that this results in v_sad_u32
    return (a > b ? a : b) - (a < b ? a : b) + c;
}

CATTR uint
OCKL_MANGLE_U32(sadhi)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_sad_hi_u8(a, b, c);
}

CATTR uint
OCKL_MANGLE_U32(sadw)(uint a, uint b, uint c)
{
    return __llvm_amdgcn_sad_u16(a, b, c);
}

CATTR float
OCKL_MANGLE_F32(unpack0)(uint a)
{
    uchar4 v = AS_UCHAR4(a);
    return (float)v.s0;
}

CATTR float
OCKL_MANGLE_F32(unpack1)(uint a)
{
    uchar4 v = AS_UCHAR4(a);
    return (float)v.s1;
}

CATTR float
OCKL_MANGLE_F32(unpack2)(uint a)
{
    uchar4 v = AS_UCHAR4(a);
    return (float)v.s2;
}

CATTR float
OCKL_MANGLE_F32(unpack3)(uint a)
{
    uchar4 v = AS_UCHAR4(a);
    return (float)v.s3;
}

