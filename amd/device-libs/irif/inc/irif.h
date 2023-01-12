/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef IRIF_H
#define IRIF_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define REQUIRES_16BIT_INSTS __attribute__((target("16-bit-insts")))
#define REQUIRES_GFX9_INSTS __attribute__((target("gfx9-insts")))
#define REQUIRES_WAVE32 __attribute__((target("wavefrontsize32")))
#define REQUIRES_WAVE64 __attribute__((target("wavefrontsize64")))

// Generic intrinsics
extern __attribute__((const)) float2 __llvm_fma_2f32(float2, float2, float2) __asm("llvm.fma.v2f32");

extern __attribute__((const)) half2 __llvm_fma_2f16(half2, half2, half2) __asm("llvm.fma.v2f16");
extern __attribute__((const)) half2 __llvm_round_2f16(half2) __asm("llvm.round.v2f16");
extern __attribute__((const)) half2 __llvm_rint_2f16(half2) __asm("llvm.rint.v2f16");

#define BUILTIN_CLZ_U8(x) (uchar)(x == 0u ? 8 : __builtin_clz(x) - 24)
#define BUILTIN_CLZ_U16(x) (ushort)(x == 0u ? 16 : __builtin_clzs(x))
#define BUILTIN_CLZ_U32(x) (uint)(x == 0u ? 32 : __builtin_clz(x))
#define BUILTIN_CLZ_U64(x) (ulong)(x == 0u ? 64 : __builtin_clzl(x))

#define BUILTIN_CTZ_U8(x) (uchar)(x == 0u ? (uchar)8 : __builtin_ctz((uint)x))
#define BUILTIN_CTZ_U16(x) (ushort)(x == 0u ? 16 : __builtin_ctzs(x))
#define BUILTIN_CTZ_U32(x) (uint)(x == 0u ? 32 : __builtin_ctz(x))
#define BUILTIN_CTZ_U64(x) (ulong)(x == 0u ? 64 : __builtin_ctzl(x))

// Buffer Load/Store
extern __attribute__((pure)) float4 __llvm_amdgcn_struct_buffer_load_format_v4f32(uint4 rsrc, uint vindex, uint voffset, uint soffset, uint cachepolicy) __asm("llvm.amdgcn.struct.buffer.load.format.v4f32");
extern __attribute__((pure)) half4 __llvm_amdgcn_struct_buffer_load_format_v4f16(uint4 rsrc, uint vindex, uint voffset, uint soffset, uint cachepolicy) __asm("llvm.amdgcn.struct.buffer.load.format.v4f16");
extern void __llvm_amdgcn_struct_buffer_store_format_v4f32(float4 vdata, uint4 rsrc, uint vindex, uint voffset, uint soffset, uint cachepolicy) __asm("llvm.amdgcn.struct.buffer.store.format.v4f32");
extern void __llvm_amdgcn_struct_buffer_store_format_v4f16( half4 vdata, uint4 rsrc, uint vindex, uint voffset, uint soffset, uint cachepolicy) __asm("llvm.amdgcn.struct.buffer.store.format.v4f16");

// Image load, store, sample, gather
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_1d_v4f32_i32(uint ix, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2d_v4f32_i32(uint ix, uint iy, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_3d_v4f32_i32(uint ix, uint iy, uint iz, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_cube_v4f32_i32(uint ix, uint iy, uint iface, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_1darray_v4f32_i32(uint ix, uint islice, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_2darray_v4f32_i32(uint ix, uint iy, uint islice, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_1d_v4f32_i32(uint ix, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_2d_v4f32_i32(uint ix, uint iy, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_3d_v4f32_i32(uint ix, uint iy, uint iz, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_cube_v4f32_i32(uint ix, uint iy, uint iface, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_1darray_v4f32_i32(uint ix, uint islice, uint imip, uint8 t);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_load_mip_2darray_v4f32_i32(uint ix, uint iy, uint islice, uint imip, uint8 t);

extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_1d_v4f16_i32(uint ix, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2d_v4f16_i32(uint ix, uint iy, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_3d_v4f16_i32(uint ix, uint iy, uint iz, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_cube_v4f16_i32(uint ix, uint iy, uint iface, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_1darray_v4f16_i32(uint ix, uint islice, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_2darray_v4f16_i32(uint ix, uint iy, uint islice, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_1d_v4f16_i32(uint ix, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_2d_v4f16_i32(uint ix, uint iy, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_3d_v4f16_i32(uint ix, uint iy, uint iz, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_cube_v4f16_i32(uint ix, uint iy, uint iface, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_1darray_v4f16_i32(uint ix, uint islice, uint imip, uint8 t);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_load_mip_2darray_v4f16_i32(uint ix, uint iy, uint islice, uint imip, uint8 t);

extern __attribute__((pure)) float __llvm_amdgcn_image_load_2d_f32_i32(uint ix, uint iy, uint8 t);
extern __attribute__((pure)) float __llvm_amdgcn_image_load_2darray_f32_i32(uint ix, uint iy, uint islice, uint8 t);
extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_2d_f32_i32(uint ix, uint iy, uint imip, uint8 t);
extern __attribute__((pure)) float __llvm_amdgcn_image_load_mip_2darray_f32_i32(uint ix, uint iy, uint islice, uint imip, uint8 t);

extern void __llvm_amdgcn_image_store_1d_v4f32_i32(float4 pix, uint ix, uint8 t);
extern void __llvm_amdgcn_image_store_2d_v4f32_i32(float4 pix, uint ix, uint iy, uint8 t);
extern void __llvm_amdgcn_image_store_3d_v4f32_i32(float4 pix, uint ix, uint iy, uint iz, uint8 t);
extern void __llvm_amdgcn_image_store_cube_v4f32_i32(float4 pix, uint ix, uint iy, uint iface, uint8 t);
extern void __llvm_amdgcn_image_store_1darray_v4f32_i32(float4 pix, uint ix, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_2darray_v4f32_i32(float4 pix, uint ix, uint iy, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1d_v4f32_i32(float4 pix, uint ix, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2d_v4f32_i32(float4 pix, uint ix, uint iy, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_3d_v4f32_i32(float4 pix, uint ix, uint iy, uint iz, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_cube_v4f32_i32(float4 pix, uint ix, uint iy, uint iface, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1darray_v4f32_i32(float4 pix, uint ix, uint islice, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2darray_v4f32_i32(float4 pix, uint ix, uint iy, uint islice, uint imip, uint8 t);

extern void __llvm_amdgcn_image_store_1d_v4f16_i32(half4 pix, uint ix, uint8 t);
extern void __llvm_amdgcn_image_store_2d_v4f16_i32(half4 pix, uint ix, uint iy, uint8 t);
extern void __llvm_amdgcn_image_store_3d_v4f16_i32(half4 pix, uint ix, uint iy, uint iz, uint8 t);
extern void __llvm_amdgcn_image_store_cube_v4f16_i32(half4 pix, uint ix, uint iy, uint iface, uint8 t);
extern void __llvm_amdgcn_image_store_1darray_v4f16_i32(half4 pix, uint ix, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_2darray_v4f16_i32(half4 pix, uint ix, uint iy, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1d_v4f16_i32(half4 pix, uint ix, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2d_v4f16_i32(half4 pix, uint ix, uint iy, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_3d_v4f16_i32(half4 pix, uint ix, uint iy, uint iz, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_cube_v4f16_i32(half4 pix, uint ix, uint iy, uint iface, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_1darray_v4f16_i32(half4 pix, uint ix, uint islice, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2darray_v4f16_i32(half4 pix, uint ix, uint iy, uint islice, uint imip, uint8 t);

extern void __llvm_amdgcn_image_store_2d_f32_i32(float pix, uint ix, uint iy, uint8 t);
extern void __llvm_amdgcn_image_store_2darray_f32_i32(float pix, uint ix, uint iy, uint islice, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2d_f32_i32(float pix, uint ix, uint iy, uint imip, uint8 t);
extern void __llvm_amdgcn_image_store_mip_2darray_f32_i32(float pix, uint ix, uint iy, uint islice, uint imip, uint8 t);

extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_1d_v4f32_f32(float x, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_1d_v4f32_f32(float x, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_1d_v4f32_f32(float x, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_1d_v4f32_f32_f32(float dxdh, float dxdv, float x, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_2d_v4f32_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_2d_v4f32_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_2d_v4f32_f32(float x, float y, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_2d_v4f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_3d_v4f32_f32(float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_3d_v4f32_f32(float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_3d_v4f32_f32(float x, float y, float z, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_3d_v4f32_f32_f32(float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_cube_v4f32_f32(float x, float y, float face, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_cube_v4f32_f32(float x, float y, float face, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_cube_v4f32_f32(float x, float y, float face, float lod, uint8 t, uint4 s);

extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_1darray_v4f32_f32(float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_1darray_v4f32_f32(float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_1darray_v4f32_f32(float x, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_1darray_v4f32_f32_f32(float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_2darray_v4f32_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_lz_2darray_v4f32_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_l_2darray_v4f32_f32(float x, float y, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_sample_d_2darray_v4f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s);

extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_1d_v4f16_f32(float x, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_1d_v4f16_f32(float x, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_1d_v4f16_f32(float x, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_1d_v4f16_f32_f32(float dxdh, float dxdv, float x, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_2d_v4f16_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_2d_v4f16_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_2d_v4f16_f32(float x, float y, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_2d_v4f16_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_3d_v4f16_f32(float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_3d_v4f16_f32(float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_3d_v4f16_f32(float x, float y, float z, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_3d_v4f16_f32_f32(float dxdh, float dydh, float dzdh, float dxdv, float dydv, float dzdv, float x, float y, float z, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_cube_v4f16_f32(float x, float y, float face, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_cube_v4f16_f32(float x, float y, float face, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_cube_v4f16_f32(float x, float y, float face, float lod, uint8 t, uint4 s);

extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_1darray_v4f16_f32(float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_1darray_v4f16_f32(float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_1darray_v4f16_f32(float x, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_1darray_v4f16_f32_f32(float dxdh, float dxdv, float x, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_2darray_v4f16_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_lz_2darray_v4f16_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_l_2darray_v4f16_f32(float x, float y, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) half4 __llvm_amdgcn_image_sample_d_2darray_v4f16_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s);

extern __attribute__((pure)) float __llvm_amdgcn_image_sample_2d_f32_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_2d_f32_f32(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_2d_f32_f32(float x, float y, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_2d_f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_2darray_f32_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_lz_2darray_f32_f32(float x, float y, float slice, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_l_2darray_f32_f32(float x, float y, float slice, float lod, uint8 t, uint4 s);
extern __attribute__((pure)) float __llvm_amdgcn_image_sample_d_2darray_f32_f32_f32(float dxdh, float dydh, float dxdv, float dydv, float x, float y, float slice, uint8 t, uint4 s);

extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_r(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_g(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_b(float x, float y, uint8 t, uint4 s);
extern __attribute__((pure)) float4 __llvm_amdgcn_image_gather4_lz_2d_v4f32_f32_a(float x, float y, uint8 t, uint4 s);


#pragma OPENCL EXTENSION cl_khr_fp16 : disable
#endif // IRIF_H
