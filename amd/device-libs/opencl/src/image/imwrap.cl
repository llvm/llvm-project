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
#pragma OPENCL EXTENSION cl_khr_mipmap_image : enable

static __constant int channel_order_map[32] = {
  CLK_A,
  CLK_R,
  CLK_Rx,
  CLK_RG,
  CLK_RGx,
  CLK_RA,
  CLK_RGB,
  CLK_RGBx,
  CLK_RGBA,
  CLK_BGRA,
  CLK_ARGB,
  666, // XXX CLK_ABGR,
  CLK_sRGB,
  CLK_sRGBx,
  CLK_sRGBA,
  CLK_sBGRA,
  CLK_INTENSITY,
  CLK_LUMINANCE,
  CLK_DEPTH,
  CLK_DEPTH_STENCIL
};

static __constant int channel_data_type_map[32] = {
  CLK_SNORM_INT8,
  CLK_SNORM_INT16,
  CLK_UNORM_INT8,
  CLK_UNORM_INT16,
  CLK_UNORM_INT24,
  CLK_UNORM_SHORT_555,
  CLK_UNORM_SHORT_565,
  CLK_UNORM_INT_101010,
  CLK_SIGNED_INT8,
  CLK_SIGNED_INT16,
  CLK_SIGNED_INT32,
  CLK_UNSIGNED_INT8,
  CLK_UNSIGNED_INT16,
  CLK_UNSIGNED_INT32,
  CLK_HALF_FLOAT,
  CLK_FLOAT
};


#define LOWER_sampler(S) __builtin_astype(S, SSHARP)

#define LOWER_ro_1D(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_1Da(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_1Db(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_2D(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_2Da(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_2Dd(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_2Dad(I) __builtin_astype(I, TSHARP)
#define LOWER_ro_3D(I) __builtin_astype(I, TSHARP)

#define LOWER_wo_1D(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_1Da(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_1Db(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_2D(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_2Da(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_2Dd(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_2Dad(I) __builtin_astype(I, TSHARP)
#define LOWER_wo_3D(I) __builtin_astype(I, TSHARP)

#define LOWER_rw_1D(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_1Da(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_1Db(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_2D(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_2Da(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_2Dd(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_2Dad(I) __builtin_astype(I, TSHARP)
#define LOWER_rw_3D(I) __builtin_astype(I, TSHARP)

#define _C(X,Y) X ## Y
#define C(X,Y) _C(X,Y)

#define PFX __ockl_image_

#define i32_fsuf i
#define u32_fsuf ui
#define f32_fsuf f
#define f16_fsuf h

#define i32_ksuf
#define u32_ksuf
#define f32_ksuf
#define f16_ksuf h

#define i32_rcast as_int4
#define u32_rcast as_uint4
#define f32_rcast
#define f16_rcast

#define _1D_ity image1d_t
#define _1Da_ity image1d_array_t
#define _1Db_ity image1d_buffer_t
#define _2D_ity image2d_t
#define _2Da_ity image2d_array_t
#define _2Dd_ity image2d_depth_t
#define _2Dad_ity image2d_array_depth_t
#define _3D_ity image3d_t

#define _1D_f32_pty float4
#define _1D_f16_pty half4
#define _1D_i32_pty int4
#define _1D_u32_pty uint4

#define _1Da_f32_pty float4
#define _1Da_f16_pty half4
#define _1Da_i32_pty int4
#define _1Da_u32_pty uint4

#define _1Db_f32_pty float4
#define _1Db_f16_pty half4
#define _1Db_i32_pty int4
#define _1Db_u32_pty uint4

#define _2D_f32_pty float4
#define _2D_f16_pty half4
#define _2D_i32_pty int4
#define _2D_u32_pty uint4

#define _2Da_f32_pty float4
#define _2Da_f16_pty half4
#define _2Da_i32_pty int4
#define _2Da_u32_pty uint4

#define _2Dd_f32_pty float

#define _2Dad_f32_pty float

#define _3D_f32_pty float4
#define _3D_f16_pty half4
#define _3D_i32_pty int4
#define _3D_u32_pty uint4

#define _1D_f32_parg p
#define _1D_f16_parg p
#define _1D_i32_parg as_float4(p)
#define _1D_u32_parg as_float4(p)

#define _1Da_f32_parg p
#define _1Da_f16_parg p
#define _1Da_i32_parg as_float4(p)
#define _1Da_u32_parg as_float4(p)

#define _1Db_f32_parg p
#define _1Db_f16_parg p
#define _1Db_i32_parg as_float4(p)
#define _1Db_u32_parg as_float4(p)

#define _2D_f32_parg p
#define _2D_f16_parg p
#define _2D_i32_parg as_float4(p)
#define _2D_u32_parg as_float4(p)

#define _2Da_f32_parg p
#define _2Da_f16_parg p
#define _2Da_i32_parg as_float4(p)
#define _2Da_u32_parg as_float4(p)

#define _2Dd_f32_parg p

#define _2Dad_f32_parg p

#define _3D_f32_parg p
#define _3D_f16_parg p
#define _3D_i32_parg as_float4(p)
#define _3D_u32_parg as_float4(p)

#define _1D_i32_cty int
#define _1D_f32_cty float

#define _1Da_i32_cty int2
#define _1Da_f32_cty float2

#define _1Db_i32_cty int

#define _2D_i32_cty int2
#define _2D_f32_cty float2

#define _2Da_i32_cty int4
#define _2Da_f32_cty float4

#define _2Dd_i32_cty int2
#define _2Dd_f32_cty float2

#define _2Dad_i32_cty int4
#define _2Dad_f32_cty float4

#define _3D_i32_cty int4
#define _3D_f32_cty float4

#define _1D_i32_carg convert_float(c)
#define _1D_f32_carg c

#define _1Da_i32_carg convert_float2(c)
#define _1Da_f32_carg c

#define _1Db_i32_carg c

#define _2D_i32_carg convert_float2(c)
#define _2D_f32_carg c

#define _2Da_i32_carg convert_float4(c)
#define _2Da_f32_carg c

#define _2Dd_i32_carg convert_float2(c)
#define _2Dd_f32_carg c

#define _2Dad_i32_carg convert_float4(c)
#define _2Dad_f32_carg c

#define _3D_i32_carg convert_float4(c)
#define _3D_f32_carg c

#define _1D_gpars float dx, float dy
#define _1Da_gpars float dx, float dy
#define _2D_gpars float2 dx, float2 dy
#define _2Da_gpars float2 dx, float2 dy
#define _2Dd_gpars float2 dx, float2 dy
#define _2Dad_gpars float2 dx, float2 dy
#define _3D_gpars float4 dx, float4 dy

#define RATTR __attribute__((overloadable, always_inline, pure))
#define WATTR __attribute__((overloadable, always_inline))
#define GATTR __attribute__((overloadable, always_inline, const))
#define FATTR __attribute__((always_inline, pure))

#define SGEN(IT,PT,CT) \
RATTR IT##_##PT##_pty \
C(read_image,PT##_fsuf)(read_only IT##_ity i, sampler_t s, IT##_##CT##_cty c) \
{ \
    return PT##_rcast(C(PFX,C(sample,C(PT##_ksuf,IT)))(LOWER_ro##IT(i), LOWER_sampler(s), IT##_##CT##_carg)); \
}

#define SGENL(IT,PT,CT) \
RATTR IT##_##PT##_pty \
C(read_image,PT##_fsuf)(read_only IT##_ity i, sampler_t s, IT##_##CT##_cty c, float l) \
{ \
    return PT##_rcast(C(PFX,C(sample,C(PT##_ksuf,C(_lod,IT))))(LOWER_ro##IT(i), LOWER_sampler(s), IT##_##CT##_carg, l)); \
}

#define SGENG(IT,PT,CT) \
RATTR IT##_##PT##_pty \
C(read_image,PT##_fsuf)(read_only IT##_ity i, sampler_t s, IT##_##CT##_cty c, IT##_gpars) \
{ \
    return PT##_rcast(C(PFX,C(sample,C(PT##_ksuf,C(_grad,IT))))(LOWER_ro##IT(i), LOWER_sampler(s), IT##_##CT##_carg, dx, dy)); \
}

#define SGENX(IT,PT,CT) \
    SGEN(IT,PT,CT) \
    SGENL(IT,PT,CT) \
    SGENG(IT,PT,CT)

#define RGEN(IT,PT,CT) \
RATTR IT##_##PT##_pty \
C(read_image,PT##_fsuf)(read_only IT##_ity i, IT##_##CT##_cty c) \
{ \
    return PT##_rcast(C(PFX,C(load,C(PT##_ksuf,IT)))(LOWER_ro##IT(i), c)); \
} \
 \
RATTR IT##_##PT##_pty \
C(read_image,PT##_fsuf)(read_write IT##_ity i, IT##_##CT##_cty c) \
{ \
    return PT##_rcast(C(PFX,C(load,C(PT##_ksuf,IT)))(LOWER_rw##IT(i), c)); \
}

#define WGEN(IT,PT,CT) \
WATTR void \
C(write_image,PT##_fsuf)(write_only IT##_ity i, IT##_##CT##_cty c, IT##_##PT##_pty p) \
{ \
    C(PFX,C(store,C(PT##_ksuf,IT)))(LOWER_wo##IT(i), c, IT##_##PT##_parg); \
} \
 \
WATTR void \
C(write_image,PT##_fsuf)(read_write IT##_ity i, IT##_##CT##_cty c, IT##_##PT##_pty p) \
{ \
    C(PFX,C(store,C(PT##_ksuf,IT)))(LOWER_rw##IT(i), c, IT##_##PT##_parg); \
}

#define WGENL(IT,PT,CT) \
WATTR void \
C(write_image,PT##_fsuf)(write_only IT##_ity i, IT##_##CT##_cty c, int l, IT##_##PT##_pty p) \
{ \
    C(PFX,C(store,C(PT##_ksuf,C(_lod,IT))))(LOWER_wo##IT(i), c, l, IT##_##PT##_parg); \
} \
 \
WATTR void \
C(write_image,PT##_fsuf)(read_write IT##_ity i, IT##_##CT##_cty c, int l, IT##_##PT##_pty p) \
{ \
    C(PFX,C(store,C(PT##_ksuf,C(_lod,IT))))(LOWER_rw##IT(i), c, l, IT##_##PT##_parg); \
}

#define WGENX(IT,PT,CT) \
    WGEN(IT,PT,CT) \
    WGENL(IT,PT,CT)

SGEN(_2D,f32,i32)
SGENX(_2D,f32,f32)
SGEN(_2D,f16,i32)
SGENX(_2D,f16,f32)
SGEN(_2D,i32,i32)
SGENX(_2D,i32,f32)
SGEN(_2D,u32,i32)
SGENX(_2D,u32,f32)

SGEN(_3D,f32,i32)
SGENX(_3D,f32,f32)
SGEN(_3D,f16,i32)
SGENX(_3D,f16,f32)
SGEN(_3D,i32,i32)
SGENX(_3D,i32,f32)
SGEN(_3D,u32,i32)
SGENX(_3D,u32,f32)

SGEN(_2Da,f32,i32)
SGENX(_2Da,f32,f32)
SGEN(_2Da,f16,i32)
SGENX(_2Da,f16,f32)
SGEN(_2Da,i32,i32)
SGENX(_2Da,i32,f32)
SGEN(_2Da,u32,i32)
SGENX(_2Da,u32,f32)

SGEN(_1D,f32,i32)
SGENX(_1D,f32,f32)
SGEN(_1D,f16,i32)
SGENX(_1D,f16,f32)
SGEN(_1D,i32,i32)
SGENX(_1D,i32,f32)
SGEN(_1D,u32,i32)
SGENX(_1D,u32,f32)

SGEN(_1Da,f32,i32)
SGENX(_1Da,f32,f32)
SGEN(_1Da,f16,i32)
SGENX(_1Da,f16,f32)
SGEN(_1Da,i32,i32)
SGENX(_1Da,i32,f32)
SGEN(_1Da,u32,i32)
SGENX(_1Da,u32,f32)

SGEN(_2Dd,f32,i32)
SGENX(_2Dd,f32,f32)

SGEN(_2Dad,f32,i32)
SGENX(_2Dad,f32,f32)

RGEN(_2D,f32,i32)
RGEN(_2D,f16,i32)
RGEN(_2D,i32,i32)
RGEN(_2D,u32,i32)

RGEN(_3D,f32,i32)
RGEN(_3D,f16,i32)
RGEN(_3D,i32,i32)
RGEN(_3D,u32,i32)

RGEN(_2Da,f32,i32)
RGEN(_2Da,f16,i32)
RGEN(_2Da,i32,i32)
RGEN(_2Da,u32,i32)

RGEN(_1D,f32,i32)
RGEN(_1D,f16,i32)
RGEN(_1D,i32,i32)
RGEN(_1D,u32,i32)

RGEN(_1Db,f32,i32)
RGEN(_1Db,f16,i32)
RGEN(_1Db,i32,i32)
RGEN(_1Db,u32,i32)

RGEN(_1Da,f32,i32)
RGEN(_1Da,f16,i32)
RGEN(_1Da,i32,i32)
RGEN(_1Da,u32,i32)

RGEN(_2Dd,f32,i32)

RGEN(_2Dad,f32,i32)

WGENX(_2D,f32,i32)
WGENX(_2D,f16,i32)
WGENX(_2D,i32,i32)
WGENX(_2D,u32,i32)

WGENX(_2Da,f32,i32)
WGENX(_2Da,f16,i32)
WGENX(_2Da,i32,i32)
WGENX(_2Da,u32,i32)

WGENX(_1D,f32,i32)
WGENX(_1D,f16,i32)
WGENX(_1D,i32,i32)
WGENX(_1D,u32,i32)

WGEN(_1Db,f32,i32)
WGEN(_1Db,f16,i32)
WGEN(_1Db,i32,i32)
WGEN(_1Db,u32,i32)

WGENX(_1Da,f32,i32)
WGENX(_1Da,f16,i32)
WGENX(_1Da,i32,i32)
WGENX(_1Da,u32,i32)

WGENX(_2Dd,f32,i32)

WGENX(_2Dad,f32,i32)

WGENX(_3D,f32,i32)
WGENX(_3D,f16,i32)
WGENX(_3D,i32,i32)
WGENX(_3D,u32,i32)


#define ro_qual read_only
#define wo_qual write_only
#define rw_qual read_write

#define GD3GEN(Q) \
GATTR int4 \
get_image_dim(Q##_qual image3d_t i) \
{ \
    return (int4)(get_image_width(i), get_image_height(i), get_image_depth(i), 0); \
}

GD3GEN(ro)
GD3GEN(wo)
GD3GEN(rw)

#define GD2GENQ(Q,T) \
GATTR int2 \
get_image_dim(Q##_qual T##_ity i) \
{ \
    return (int2)(get_image_width(i), get_image_height(i)); \
}

#define GD2GEN(T) \
    GD2GENQ(ro,T) \
    GD2GENQ(wo,T) \
    GD2GENQ(rw,T)

GD2GEN(_2D)
GD2GEN(_2Da)
GD2GEN(_2Dd)
GD2GEN(_2Dad)

#define GGENQT(Q,N,T) \
GATTR int \
get_image_##N(Q##_qual T##_ity i) { \
    return C(PFX,C(N,T))(LOWER_##Q##T(i)); \
}

#define GGENT(N,T) \
    GGENQT(ro,N,T) \
    GGENQT(wo,N,T) \
    GGENQT(rw,N,T)

#define GGENX(N) \
    GGENT(N,_1D) \
    GGENT(N,_1Da) \
    GGENT(N,_2D) \
    GGENT(N,_2Da) \
    GGENT(N,_2Dd) \
    GGENT(N,_2Dad) \
    GGENT(N,_3D)

#define GGEN(N) \
    GGENX(N) \
    GGENT(N,_1Db) \

GGEN(width)
GGENX(num_mip_levels)

// int get depth _3D
#define GNZGEN(Q) \
GATTR int \
get_image_depth(Q##_qual image3d_t i) \
{ \
    return C(PFX,depth_3D)(LOWER_##Q##_3D(i)); \
}

GNZGEN(ro)
GNZGEN(wo)
GNZGEN(rw)

// size_t get image_array_size _1Da, _2Da, _2Dad
#define GASGENQ(Q,T) \
GATTR size_t \
get_image_array_size(Q##_qual T##_ity i) \
{ \
    return C(PFX,C(array_size,T))(LOWER_##Q##T(i)); \
}

#define GASGEN(T) \
    GASGENQ(ro,T) \
    GASGENQ(wo,T) \
    GASGENQ(rw,T)

GASGEN(_1Da)
GASGEN(_2Da)
GASGEN(_2Dad)

#define GCOGENQ(Q,T) \
GATTR int \
get_image_channel_order(Q##_qual T##_ity i) { \
    return channel_order_map[C(PFX,C(channel_order,T))(LOWER_##Q##T(i))]; \
}

#define GCOGEN(T) \
    GCOGENQ(ro,T) \
    GCOGENQ(wo,T) \
    GCOGENQ(rw,T)

GCOGEN(_1D)
GCOGEN(_1Da)
GCOGEN(_1Db)
GCOGEN(_2D)
GCOGEN(_2Da)
GCOGEN(_2Dd)
GCOGEN(_2Dad)
GCOGEN(_3D)

#define GDTGENQ(Q,T) \
GATTR int \
get_image_channel_data_type(Q##_qual T##_ity i) { \
    return channel_data_type_map[C(PFX,C(channel_data_type,T))(LOWER_##Q##T(i))]; \
}

#define GDTGEN(T) \
    GDTGENQ(ro,T) \
    GDTGENQ(wo,T) \
    GDTGENQ(rw,T)

GDTGEN(_1D)
GDTGEN(_1Da)
GDTGEN(_1Db)
GDTGEN(_2D)
GDTGEN(_2Da)
GDTGEN(_2Dd)
GDTGEN(_2Dad)
GDTGEN(_3D)

#define GNYGENQ(Q,T) \
GATTR int \
get_image_height(Q##_qual T##_ity i) { \
    return C(PFX,C(height,T))(LOWER_##Q##T(i)); \
}

#define GNYGEN(T) \
    GNYGENQ(ro,T) \
    GNYGENQ(wo,T) \
    GNYGENQ(rw,T)

GNYGEN(_2D)
GNYGEN(_2Da)
GNYGEN(_2Dd)
GNYGEN(_2Dad)
GNYGEN(_3D)

FATTR float4
amd_fetch4_ff(read_only image2d_t im, float2 coord, int comp)
{
    sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
    switch (comp) {
    case 1:  return __ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    case 2:  return __ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    case 3:  return __ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    default: return __ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    }
}

FATTR float4
amd_fetch4_fsf(read_only image2d_t im, sampler_t s, float2 coord, int comp)
{
    switch (comp) {
    case 1:  return __ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    case 2:  return __ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    case 3:  return __ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    default: return __ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord);
    }
}

FATTR float4
amd_fetch4_fi(read_only image2d_t im, int2 coord, int comp)
{
    sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
    float2 fcoord = convert_float2(coord);
    switch (comp) {
    case 1:  return __ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    case 2:  return __ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    case 3:  return __ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    default: return __ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    }
}

FATTR float4
amd_fetch4_fsi(read_only image2d_t im, sampler_t s, int2 coord, int comp)
{
    float2 fcoord = convert_float2(coord);
    switch (comp) {
    case 1:  return __ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    case 2:  return __ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    case 3:  return __ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    default: return __ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord);
    }
}

FATTR int4
amd_fetch4_if(read_only image2d_t im, float2 coord, int comp)
{
    sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
    if (__oclc_ISA_version() < 900) {
        coord -= 0.5f;
    }
    switch (comp) {
    case 1:  return as_int4(__ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    case 2:  return as_int4(__ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    case 3:  return as_int4(__ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    default: return as_int4(__ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    }
}

FATTR int4
amd_fetch4_isf(read_only image2d_t im, sampler_t s, float2 coord, int comp)
{
    if (__oclc_ISA_version() < 900) {
        coord -= 0.5f;
    }
    switch (comp) {
    case 1:  return as_int4(__ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    case 2:  return as_int4(__ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    case 3:  return as_int4(__ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    default: return as_int4(__ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), coord));
    }
}

FATTR int4
amd_fetch4_ii(read_only image2d_t im, int2 coord, int comp)
{
    sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_NONE;
    float2 fcoord = convert_float2(coord);
    if (__oclc_ISA_version() < 900) {
        fcoord -= 0.5f;
    }
    switch (comp) {
    case 1:  return as_int4(__ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    case 2:  return as_int4(__ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    case 3:  return as_int4(__ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    default: return as_int4(__ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    }
}

FATTR int4
amd_fetch4_isi(read_only image2d_t im, sampler_t s, int2 coord, int comp)
{
    float2 fcoord = convert_float2(coord);
    if (__oclc_ISA_version() < 900) {
        fcoord -= 0.5f;
    }
    switch (comp) {
    case 1:  return as_int4(__ockl_image_gather4g_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    case 2:  return as_int4(__ockl_image_gather4b_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    case 3:  return as_int4(__ockl_image_gather4a_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    default: return as_int4(__ockl_image_gather4r_2D(LOWER_ro_2D(im), LOWER_sampler(s), fcoord));
    }
}

FATTR uint4
amd_fetch4_uf(read_only image2d_t im, float2 coord, int comp)
{
    return as_uint4(amd_fetch4_if(im, coord, comp));
}

FATTR uint4
amd_fetch4_usf(read_only image2d_t im, sampler_t s, float2 coord, int comp)
{
    return as_uint4(amd_fetch4_isf(im, s, coord, comp));
}

FATTR uint4
amd_fetch4_ui(read_only image2d_t im, int2 coord, int comp)
{
    return as_uint4(amd_fetch4_ii(im, coord, comp));
}

FATTR uint4
amd_fetch4_usi(read_only image2d_t im, sampler_t s, int2 coord, int comp)
{
    return as_uint4(amd_fetch4_isi(im, s, coord, comp));
}

