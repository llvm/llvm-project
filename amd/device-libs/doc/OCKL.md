# OCKL User Guide

* [Introduction](#introduction)
  * [What Is OCKL](#what-is-ockl)
* [Using OCKL](#using-ocml)
  * [Standard Usage](#standard-usage)
  * [Controls](#controls)
* [Versioning](#versioning)
* [Naming convention](#naming-convention)
* [Supported functions](#supported-functions)


## Introduction
### What Is OCKL

OCKL is an LLVM-IR bitcode library designed to provide access to certain hardware
and compiler capabilities needed by language runtimes.  It should rarely be necessary
to call any of these functions directly from application code.  Consider this library
a "detail" layer.

## Using OCKL
### Standard Usage

OCKL is expected to be used in a standard LLVM compilation flow as follows:
  * Compile source modules to LLVM-IR bitcode (clang)
  * Link together program bitcode with library bitcode including OCKL and OCLC.
  * Run generic optimizations (opt)
  * Code generation (llc)

### Controls

OCKL supports a number of controls that are provided by linking in specifically named inline
functions.  These functions are inlined at optimization time and result in specific paths
taken with no control flow overhead.  These functions all have the form (in C)

    __attribute__((always_inline, const)) int
    __oclc_control(void)
    { return 1; } // or 0 to disable

The currently supported control are
  * `finite_only_opt` - floating point Inf and NaN are never expected to be consumed or produced
  * `unsafe_math_opt` - lower accuracy results may be produced with higher performance
  * `daz_opt` - subnormal values consumed and produced may be flushed to zero
  * `correctly_rounded_sqrt32` - float square root must be correctly rounded
  * `ISA_version` - an integer representation of the ISA version of the target device

### Versioning

OCKL usually ships as a single LLVM-IR bitcode file named

    ocml-{LLVM rev}-{OCKL rev}.bc

where `{LLVM rev}` is the version of LLVM used to create the file, of the
form X.Y, e.g. 3.8, and `{OCKL rev}` is the OCKL library version of the form X.Y, currently 0.9.

### Naming convention

OCKL functions follow a simple naming convention:

    __ockl_{function}_{type suffix}

where {type suffix} generally indicates the type of the arguments and/or returned result using a type letter,
e.g. "u" for unsigned integer, and a bit width, e.g. 32.

### Supported functions

The following table lists the available functions along with a brief description of each:

| **function** | **Brief Description** |
| :--- | :--- |
| `uchar __ockl_clz_u8(uchar);` | Count leading zeroes |
| `ushort __ockl_clz_u16(ushort);` | |
| `uint __ockl_clz_u32(uint);` | |
| `ulong __ockl_clz_u64(ulong);` | |
| - | |
| `uchar __ockl_ctz_u8(uchar);` | Count trailing zeroes |
| `ushort __ockl_ctz_u16(ushort);` | |
| `uint __ockl_ctz_u32(uint);` | |
| `ulong __ockl_ctz_u64(ulong);` | |
| - | |
| `uint __ockl_popcount_u32(uint);` | Count nonzero bits |
| `ulong __ockl_popcount_u64(ulong);` | |
| - | |
| `int __ockl_add_sat_i32(int,int);` | Add with saturation |
| `uint __ockl_add_sat_u32(uint,uint);` | |
| `long __ockl_add_sat_i64(long,long);` | |
| `ulong __ockl_add_sat_u64(ulong,ulong);` | |
| - | |
| `int __ockl_sub_sat_i32(int,int);` | Subtract with saturation |
| `uint __ockl_sub_sat_u32(uint,uint);` | |
| `long __ockl_sub_sat_i64(long,long);` | |
| `ulong __ockl_sub_sat_u64(ulong,ulong);` | |
| - | |
| `int __ockl_mul_hi_i32(int,int);` | High part of multiplication |
| `uint __ockl_mul_hi_u32(uint,uint);` | |
| `long __ockl_mul_hi_i64(long,long);` | |
| `ulong __ockl_mul_hi_u64(ulong,ulong);` | |
| - | |
| `int __ockl_mul24_i32(int,int);` | Multiply assuming operands fit in 24 bits |
| `uint __ockl_mul24_u32(uint,uint);` | |
| - | |
| `ulong __ockl_cyclectr_u64(void);` | Current value of free running 64-bit clock counter |
| `ulong __ockl_steadyctr_u64(void);` | Current value of constant speed 64-bit clock counter |
| - | |
| `uint __ockl_activelane_u32(void);` | Index of currently lane counting only active lanes in wavefront |
| - | |
| `half __ockl_wfred_add_f16(half x);` | ADD reduction across wavefront |
| `float __ockl_wfred_add_f32(float x);` | |
| `double __ockl_wfred_add_f64(double x);` | |
| `int __ockl_wfred_add_i32(int x);` | |
| `long __ockl_wfred_add_i64(long x);` | |
| `uint __ockl_wfred_add_u32(uint x);` | |
| `ulong __ockl_wfred_add_u64(ulong x);` | AND reduction across wavefront |
| `int __ockl_wfred_and_i32(int x);` | |
| `long __ockl_wfred_and_i64(long x);` | |
| `uint __ockl_wfred_and_u32(uint x);` | |
| `ulong __ockl_wfred_and_u64(ulong x);` | |
| `half __ockl_wfred_max_f16(half x);` | MAX reduction across wavefront |
| `float __ockl_wfred_max_f32(float x);` | |
| `double __ockl_wfred_max_f64(double x);` | |
| `int __ockl_wfred_max_i32(int x);` | |
| `long __ockl_wfred_max_i64(long x);` | |
| `uint __ockl_wfred_max_u32(uint x);` | |
| `ulong __ockl_wfred_max_u64(ulong x);` | |
| `half __ockl_wfred_min_f16(half x);` | MIN reduction across wavefront |
| `float __ockl_wfred_min_f32(float x);` | |
| `double __ockl_wfred_min_f64(double x);` | |
| `int __ockl_wfred_min_i32(int x);` | |
| `long __ockl_wfred_min_i64(long x);` | |
| `uint __ockl_wfred_min_u32(uint x);` | |
| `ulong __ockl_wfred_min_u64(ulong x);` | |
| `int __ockl_wfred_or_i32(int x);` | OR reduction across wavefront |
| `long __ockl_wfred_or_i64(long x);` | |
| `uint __ockl_wfred_or_u32(uint x);` | |
| `ulong __ockl_wfred_or_u64(ulong x);` | |
| `int __ockl_wfred_xor_i32(int x);` | XOR reduction across wavefront |
| `long __ockl_wfred_xor_i64(long x);` | |
| `uint __ockl_wfred_xor_u32(uint x);` | |
| `ulong __ockl_wfred_xor_u64(ulong x);` | |
| `half __ockl_wfscan_add_f16(half x, bool inclusive);` | ADD scan across wavefront |
| `float __ockl_wfscan_add_f32(float x, bool inclusive);` | |
| `double __ockl_wfscan_add_f64(double x, bool inclusive);` | |
| `int __ockl_wfscan_add_i32(int x, bool inclusive);` | |
| `long __ockl_wfscan_add_i64(long x, bool inclusive);` | |
| `uint __ockl_wfscan_add_u32(uint x, bool inclusive);` | |
| `ulong __ockl_wfscan_add_u64(ulong x, bool inclusive);` | |
| `int __ockl_wfscan_and_i32(int x, bool inclusive);` | AND scan across wavefront |
| `long __ockl_wfscan_and_i64(long x, bool inclusive);` | |
| `uint __ockl_wfscan_and_u32(uint x, bool inclusive);` | |
| `ulong __ockl_wfscan_and_u64(ulong x, bool inclusive);` | |
| `half __ockl_wfscan_max_f16(half x, bool inclusive);` | MAX scan across wavefront |
| `float __ockl_wfscan_max_f32(float x, bool inclusive);` | |
| `double __ockl_wfscan_max_f64(double x, bool inclusive);` | |
| `int __ockl_wfscan_max_i32(int x, bool inclusive);` | |
| `long __ockl_wfscan_max_i64(long x, bool inclusive);` | |
| `uint __ockl_wfscan_max_u32(uint x, bool inclusive);` | |
| `ulong __ockl_wfscan_max_u64(ulong x, bool inclusive);` | |
| `half __ockl_wfscan_min_f16(half x, bool inclusive);` | MIN scan across wavefront |
| `float __ockl_wfscan_min_f32(float x, bool inclusive);` | |
| `double __ockl_wfscan_min_f64(double x, bool inclusive);` | |
| `int __ockl_wfscan_min_i32(int x, bool inclusive);` | |
| `long __ockl_wfscan_min_i64(long x, bool inclusive);` | |
| `uint __ockl_wfscan_min_u32(uint x, bool inclusive);` | |
| `ulong __ockl_wfscan_min_u64(ulong x, bool inclusive);` | |
| `int __ockl_wfscan_or_i32(int x, bool inclusive);` | OR scan across wavefront |
| `long __ockl_wfscan_or_i64(long x, bool inclusive);` | |
| `uint __ockl_wfscan_or_u32(uint x, bool inclusive);` | |
| `ulong __ockl_wfscan_or_u64(ulong x, bool inclusive);` | |
| `int __ockl_wfscan_xor_i32(int x, bool inclusive);` | XOR scan across wavefront |
| `long __ockl_wfscan_xor_i64(long x, bool inclusive);` | |
| `uint __ockl_wfscan_xor_u32(uint x, bool inclusive);` | |
| `ulong __ockl_wfscan_xor_u64(ulong x, bool inclusive);` | |
| `uint __ockl_wfbcast_u32(uint x, uint i);` | Broadcast to wavefront |
| `ulong __ockl_wfbcast_u64(ulong x, uint i);` | |
| - | |
| `bool __ockl_wfany_i32(int e);` | Detect any nonzero across wavefront |
| `bool __ockl_wfall_i32(int e);` | Detect all nozero across wavefront |
| `bool __ockl_wfsame_i32(int e);` | Detect same across wavefront  |
| - | |
| `uint __ockl_bfm_u32(uint,uint);` | Bit field mask |
| `int __ockl_bfe_i32(int, uint, uint);` | Bit field extract |
| `uint __ockl_bfe_u32(uint,uint,uint);` | |
| `uint __ockl_bitalign_u32(uint,uint,uint);` | Align on bit boundary |
| `uint __ockl_bytealign_u32(uint,uint,uint);` | Align on byte boundary |
| `uint __ockl_lerp_u32(uint,uint,uint);` | Add each byte with prescribed carry |
| `float __ockl_max3_f32(float,float,float);` | Max of 3 |
| `half __ockl_max3_f16(half,half,half);` | |
| `int __ockl_max3_i32(int,int,int);` | |
| `uint __ockl_max3_u32(uint,uint,uint);` | |
| `float __ockl_median3_f32(float,float,float);` | Median of 3 |
| `half __ockl_median3_f16(half,half,half);` | |
| `int __ockl_median3_i32(int,int,int);` | |
| `uint __ockl_median3_u32(uint,uint,uint);` | |
| `float __ockl_min3_f32(float,float,float);` | Min of 3 |
| `half __ockl_min3_f16(half,half,half);` | |
| `int __ockl_min3_i32(int,int,int);` | |
| `uint __ockl_min3_u32(uint,uint,uint);` | |
| `ulong __ockl_mqsad_u64(ulong, uint, ulong);` | Masked rolling SAD |
| `uint __ockl_pack_u32(float4);` | Pack vector to bytes |
| `ulong __ockl_qsad_u64(ulong, uint, ulong);` | Rolling SAD |
| `uint __ockl_msad_u32(uint,uint,uint);` | Masked SAD |
| `uint __ockl_sad_u32(uint,uint,uint);` | SAD |
| `uint __ockl_sadd_u32(uint,uint,uint);` | 32-bit SAD |
| `uint __ockl_sadhi_u32(uint,uint,uint);` | SAD accululating to high half |
| `uint __ockl_sadw_u32(uint,uint,uint);` | 16-bit SAD |
| `float __ockl_unpack0_f32(uint);` | Extract byte and convert to float |
| `float __ockl_unpack1_f32(uint);` | |
| `float __ockl_unpack2_f32(uint);` | |
| `float __ockl_unpack3_f32(uint);` | |
| - | |
| `float4 __ockl_image_load_1D(TSHARP i, int c);` | Load from 1D image |
| `float4 __ockl_image_load_1Da(TSHARP i, int2 c);` | Load from 1D image array |
| `float4 __ockl_image_load_1Db(TSHARP i, int c);` | Load from 1D buffered image |
| `float4 __ockl_image_load_2D(TSHARP i, int2 c);` | Load from 2D image |
| `float4 __ockl_image_load_2Da(TSHARP i, int4 c);` | Load from 2D image array |
| `float __ockl_image_load_2Dad(TSHARP i, int4 c);` | Load from 2D depth image array |
| `float __ockl_image_load_2Dd(TSHARP i, int2 c);` | Load from 2D depth image |
| `float4 __ockl_image_load_3D(TSHARP i, int4 c);` | Load from 3D image |
| `float4 __ockl_image_load_CM(TSHARP i, int2 c, int f);` | Load from cubemap |
| `float4 __ockl_image_load_CMa(TSHARP i, int4 c, int f);` | Load from cubemap array |
| - | |
| `float4 __ockl_image_load_mip_1D(TSHARP i, int c, int l);` | Load from mipmapped image |
| `float4 __ockl_image_load_mip_1Da(TSHARP i, int2 c, int l);` | |
| `float4 __ockl_image_load_mip_2D(TSHARP i, int2 c, int l);` | |
| `float4 __ockl_image_load_mip_2Da(TSHARP i, int4 c, int l);` | |
| `float __ockl_image_load_mip_2Dad(TSHARP i, int4 c, int l);` | |
| `float __ockl_image_load_mip_2Dd(TSHARP i, int2 c, int l);` | |
| `float4 __ockl_image_load_mip_3D(TSHARP i, int4 c, int l);` | |
| `float4 __ockl_image_load_mip_CM(TSHARP i, int2 c, int f, int l);` | |
| `float4 __ockl_image_load_mip_CMa(TSHARP i, int4 c, int f, int l);` | |
| - | |
| `half4 __ockl_image_loadh_1D(TSHARP i, int c);` | Load from image returning half precision |
| `half4 __ockl_image_loadh_1Da(TSHARP i, int2 c);` | |
| `half4 __ockl_image_loadh_1Db(TSHARP i, int c);` | |
| `half4 __ockl_image_loadh_2D(TSHARP i, int2 c);` | |
| `half4 __ockl_image_loadh_2Da(TSHARP i, int4 c);` | |
| `half4 __ockl_image_loadh_3D(TSHARP i, int4 c);` | |
| `half4 __ockl_image_loadh_CM(TSHARP i, int2 c, int f);` | |
| `half4 __ockl_image_loadh_CMa(TSHARP i, int4 c, int f);` | |
| `half4 __ockl_image_loadh_mip_1D(TSHARP i, int c, int l);` | |
| `half4 __ockl_image_loadh_mip_1Da(TSHARP i, int2 c, int l);` | |
| `half4 __ockl_image_loadh_mip_2D(TSHARP i, int2 c, int l);` | |
| `half4 __ockl_image_loadh_mip_2Da(TSHARP i, int4 c, int l);` | |
| `half4 __ockl_image_loadh_mip_3D(TSHARP i, int4 c, int l);` | |
| `half4 __ockl_image_loadh_mip_CM(TSHARP i, int2 c, int f, int l);` | |
| `half4 __ockl_image_loadh_mip_CMa(TSHARP i, int4 c, int f, int l);` | |
| - | |
| `void __ockl_image_store_1D(TSHARP i, int c, float4 p);` | Store to image |
| `void __ockl_image_store_1Da(TSHARP i, int2 c, float4 p);` | |
| `void __ockl_image_store_1Db(TSHARP i, int c, float4 p);` | |
| `void __ockl_image_store_2D(TSHARP i, int2 c, float4 p);` | |
| `void __ockl_image_store_2Da(TSHARP i, int4 c, float4 p);` | |
| `void __ockl_image_store_2Dad(TSHARP i, int4 c, float p);` | |
| `void __ockl_image_store_2Dd(TSHARP i, int2 c, float p);` | |
| `void __ockl_image_store_3D(TSHARP i, int4 c, float4 p);` | |
| `void __ockl_image_store_CM(TSHARP i, int2 c, int f, float4 p);` | |
| `void __ockl_image_store_CMa(TSHARP i, int4 c, int f, float4 p);` | |
| `void __ockl_image_store_lod_1D(TSHARP i, int c, int l, float4 p);` | Store to level of mipmapped image |
| - | |
| `void __ockl_image_store_lod_1Da(TSHARP i, int2 c, int l, float4 p);` | |
| `void __ockl_image_store_lod_2D(TSHARP i, int2 c, int l, float4 p);` | |
| `void __ockl_image_store_lod_2Da(TSHARP i, int4 c, int l, float4 p);` | |
| `void __ockl_image_store_lod_2Dad(TSHARP i, int4 c, int l, float p);` | |
| `void __ockl_image_store_lod_2Dd(TSHARP i, int2 c, int l, float p);` | |
| `void __ockl_image_store_lod_3D(TSHARP i, int4 c, int l, float4 p);` | |
| `void __ockl_image_store_lod_CM(TSHARP i, int2 c, int f, int l, float4 p);` | |
| `void __ockl_image_store_lod_CMa(TSHARP i, int4 c, int f, int l, float4 p);` | |
| - | |
| `void __ockl_image_storeh_1D(TSHARP i, int c, half4 p);` | Store half precision pixel to image|
| `void __ockl_image_storeh_1Da(TSHARP i, int2 c, half4 p);` | |
| `void __ockl_image_storeh_1Db(TSHARP i, int c, half4 p);` | |
| `void __ockl_image_storeh_2D(TSHARP i, int2 c, half4 p);` | |
| `void __ockl_image_storeh_2Da(TSHARP i, int4 c, half4 p);` | |
| `void __ockl_image_storeh_3D(TSHARP i, int4 c, half4 p);` | |
| `void __ockl_image_storeh_CM(TSHARP i, int2 c, int f, half4 p);` | |
| `void __ockl_image_storeh_CMa(TSHARP i, int4 c, int f, half4 p);` | |
| - | |
| `void __ockl_image_storeh_lod_1D(TSHARP i, int c, int l, half4 p);` | Store half precision pixel to level of mipmapped image |
| `void __ockl_image_storeh_lod_1Da(TSHARP i, int2 c, int l, half4 p);` | |
| `void __ockl_image_storeh_lod_2D(TSHARP i, int2 c, int l, half4 p);` | |
| `void __ockl_image_storeh_lod_2Da(TSHARP i, int4 c, int l, half4 p);` | |
| `void __ockl_image_storeh_lod_3D(TSHARP i, int4 c, int l, half4 p);` | |
| `void __ockl_image_storeh_lod_CM(TSHARP i, int2 c, int f, int l, half4 p);` | |
| `void __ockl_image_storeh_lod_CMa(TSHARP i, int4 c, int f, int l, half4 p);` | |
| - | |
| `float4 __ockl_image_sample_1D(TSHARP i, SSHARP s, float c);` | Sample image |
| `float4 __ockl_image_sample_1Da(TSHARP i, SSHARP s, float2 c);` | |
| `float4 __ockl_image_sample_2D(TSHARP i, SSHARP s, float2 c);` | |
| `float4 __ockl_image_sample_2Da(TSHARP i, SSHARP s, float4 c);` | |
| `float __ockl_image_sample_2Dad(TSHARP i, SSHARP s, float4 c);` | |
| `float __ockl_image_sample_2Dd(TSHARP i, SSHARP s, float2 c);` | |
| `float4 __ockl_image_sample_3D(TSHARP i, SSHARP s, float4 c);` | |
| `float4 __ockl_image_sample_CM(TSHARP i, SSHARP s, float4 c);` | |
| `float4 __ockl_image_sample_CMa(TSHARP i, SSHARP s, float4 c);` | |
| - | |
| `float4 __ockl_image_sample_grad_1D(TSHARP i, SSHARP s, float c, float dx, float dy);` | Sample mipmapped image using gradient |
| `float4 __ockl_image_sample_grad_1Da(TSHARP i, SSHARP s, float2 c, float dx, float dy);` | |
| `float4 __ockl_image_sample_grad_2D(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy);` | |
| `float4 __ockl_image_sample_grad_2Da(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy);` | |
| `float __ockl_image_sample_grad_2Dad(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy);` | |
| `float __ockl_image_sample_grad_2Dd(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy);` | |
| `float4 __ockl_image_sample_grad_3D(TSHARP i, SSHARP s, float4 c, float4 dx, float4 dy);` | |
| - | |
| `float4 __ockl_image_sample_lod_1D(TSHARP i, SSHARP s, float c, float l);` | Sample mipmapped image using LOD |
| `float4 __ockl_image_sample_lod_1Da(TSHARP i, SSHARP s, float2 c, float l);` | |
| `float4 __ockl_image_sample_lod_2D(TSHARP i, SSHARP s, float2 c, float l);` | |
| `float4 __ockl_image_sample_lod_2Da(TSHARP i, SSHARP s, float4 c, float l);` | |
| `float __ockl_image_sample_lod_2Dad(TSHARP i, SSHARP s, float4 c, float l);` | |
| `float __ockl_image_sample_lod_2Dd(TSHARP i, SSHARP s, float2 c, float l);` | |
| `float4 __ockl_image_sample_lod_3D(TSHARP i, SSHARP s, float4 c, float l);` | |
| `float4 __ockl_image_sample_lod_CM(TSHARP i, SSHARP s, float4 c, float l);` | |
| `float4 __ockl_image_sample_lod_CMa(TSHARP i, SSHARP s, float4 c, float l);` | |
| - | |
| `half4 __ockl_image_sampleh_1D(TSHARP i, SSHARP s, float c);` | Sample image returning half precision |
| `half4 __ockl_image_sampleh_1Da(TSHARP i, SSHARP s, float2 c);` | |
| `half4 __ockl_image_sampleh_2D(TSHARP i, SSHARP s, float2 c);` | |
| `half4 __ockl_image_sampleh_2Da(TSHARP i, SSHARP s, float4 c);` | |
| `half4 __ockl_image_sampleh_3D(TSHARP i, SSHARP s, float4 c);` | |
| `half4 __ockl_image_sampleh_CM(TSHARP i, SSHARP s, float4 c);` | |
| `half4 __ockl_image_sampleh_CMa(TSHARP i, SSHARP s, float4 c);` | |
| - | |
| `half4 __ockl_image_sampleh_grad_1D(TSHARP i, SSHARP s, float c, float dx, float dy);` | Sample mipmapped image using gradient returning half precision |
| `half4 __ockl_image_sampleh_grad_1Da(TSHARP i, SSHARP s, float2 c, float dx, float dy);` | |
| `half4 __ockl_image_sampleh_grad_2D(TSHARP i, SSHARP s, float2 c, float2 dx, float2 dy);` | |
| `half4 __ockl_image_sampleh_grad_2Da(TSHARP i, SSHARP s, float4 c, float2 dx, float2 dy);` | |
| `half4 __ockl_image_sampleh_grad_3D(TSHARP i, SSHARP s, float4 c, float4 dx, float4 dy);` | |
| - | |
| `half4 __ockl_image_sampleh_lod_1D(TSHARP i, SSHARP s, float c, float l);` | Sample mipmapped image using LOD returning half precision |
| `half4 __ockl_image_sampleh_lod_1Da(TSHARP i, SSHARP s, float2 c, float l);` | |
| `half4 __ockl_image_sampleh_lod_2D(TSHARP i, SSHARP s, float2 c, float l);` | |
| `half4 __ockl_image_sampleh_lod_2Da(TSHARP i, SSHARP s, float4 c, float l);` | |
| `half4 __ockl_image_sampleh_lod_3D(TSHARP i, SSHARP s, float4 c, float l);` | |
| `half4 __ockl_image_sampleh_lod_CM(TSHARP i, SSHARP s, float4 c, float l);` | |
| `half4 __ockl_image_sampleh_lod_CMa(TSHARP i, SSHARP s, float4 c, float l);` | |
| - | |
| `float4 __ockl_image_gather4r_2D(TSHARP i, SSHARP s, float2 c);` | Gather 2x2 channel from image |
| `float4 __ockl_image_gather4g_2D(TSHARP i, SSHARP s, float2 c);` | |
| `float4 __ockl_image_gather4b_2D(TSHARP i, SSHARP s, float2 c);` | |
| `float4 __ockl_image_gather4a_2D(TSHARP i, SSHARP s, float2 c);` | |
| - | |
| `int __ockl_image_array_size_1Da(TSHARP i);` | Get image array size |
| `int __ockl_image_array_size_2Da(TSHARP i);` | |
| `int __ockl_image_array_size_2Dad(TSHARP i);` | |
| `int __ockl_image_array_size_CMa(TSHARP i);` | |
| - | |
| `int __ockl_image_channel_data_type_1D(TSHARP i);` | Get image channel data type |
| `int __ockl_image_channel_data_type_1Da(TSHARP i);` | |
| `int __ockl_image_channel_data_type_1Db(TSHARP i);` | |
| `int __ockl_image_channel_data_type_2D(TSHARP i);` | |
| `int __ockl_image_channel_data_type_2Da(TSHARP i);` | |
| `int __ockl_image_channel_data_type_2Dad(TSHARP i);` | |
| `int __ockl_image_channel_data_type_2Dd(TSHARP i);` | |
| `int __ockl_image_channel_data_type_3D(TSHARP i);` | |
| `int __ockl_image_channel_data_type_CM(TSHARP i);` | |
| `int __ockl_image_channel_data_type_CMa(TSHARP i);` | |
| - | |
| `int __ockl_image_channel_order_1D(TSHARP i);` | Get image channel order |
| `int __ockl_image_channel_order_1Da(TSHARP i);` | |
| `int __ockl_image_channel_order_1Db(TSHARP i);` | |
| `int __ockl_image_channel_order_2D(TSHARP i);` | |
| `int __ockl_image_channel_order_2Da(TSHARP i);` | |
| `int __ockl_image_channel_order_2Dad(TSHARP i);` | |
| `int __ockl_image_channel_order_2Dd(TSHARP i);` | |
| `int __ockl_image_channel_order_3D(TSHARP i);` | |
| `int __ockl_image_channel_order_CM(TSHARP i);` | |
| `int __ockl_image_channel_order_CMa(TSHARP i);` | |
| - | |
| `int __ockl_image_depth_3D(TSHARP i);` | Get 3D image depth |
| - | |
| `int __ockl_image_height_2D(TSHARP i);` | Get image height |
| `int __ockl_image_height_2Da(TSHARP i);` | |
| `int __ockl_image_height_2Dad(TSHARP i);` | |
| `int __ockl_image_height_2Dd(TSHARP i);` | |
| `int __ockl_image_height_3D(TSHARP i);` | |
| `int __ockl_image_height_CM(TSHARP i);` | |
| `int __ockl_image_height_CMa(TSHARP i);` | |
| - | |
| `int __ockl_image_num_mip_levels_1D(TSHARP i);` | Get number of levels in mipmapped image |
| `int __ockl_image_num_mip_levels_1Da(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_2D(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_2Da(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_2Dad(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_2Dd(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_3D(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_CM(TSHARP i);` | |
| `int __ockl_image_num_mip_levels_CMa(TSHARP i);` | |
| - | |
| `int __ockl_image_width_1D(TSHARP i);` | Get image width |
| `int __ockl_image_width_1Da(TSHARP i);` | |
| `int __ockl_image_width_1Db(TSHARP i);` | |
| `int __ockl_image_width_2D(TSHARP i);` | |
| `int __ockl_image_width_2Da(TSHARP i);` | |
| `int __ockl_image_width_2Dad(TSHARP i);` | |
| `int __ockl_image_width_2Dd(TSHARP i);` | |
| `int __ockl_image_width_3D(TSHARP i);` | |
| `int __ockl_image_width_CM(TSHARP i);` | |
| `int __ockl_image_width_CMa(TSHARP i);` | |
| - | |
| `size_t __ockl_get_global_offset(uint);` | Get grid global offset (OpenCL) of dimension |
| `size_t __ockl_get_global_id(uint);` | Get workitem global ID of dimension |
| `size_t __ockl_get_local_id(uint);` | Get workitem local ID of dimension |
| `size_t __ockl_get_group_id(uint);` | Get ID of group workitem resides in of dimension |
| `size_t __ockl_get_global_size(uint);` | Get global size of dimension |
| `size_t __ockl_get_local_size(uint);` | Get local size of dimension |
| `size_t __ockl_get_num_groups(uint);` | Get number of groups in dimension |
| `uint __ockl_get_work_dim(void);` | Get grid number of dimensions |
| `size_t __ockl_get_enqueued_local_size(uint);` | Get enqueued local size of dimension |
| `size_t __ockl_get_global_linear_id(void);` | Get global linear ID of workitem|
| `size_t __ockl_get_local_linear_id(void);` | Get local linear ID of workitem |
| - | |
| `bool __ockl_is_local_addr(const void *);` | Test if generic address is local |
| `bool __ockl_is_private_addr(const void *);` | Test if generic address is private |
| `__global void * __ockl_to_global(void *);` | Convert generic address to global address |
| `__local void * __ockl_to_local(void *);` | Convert generic address to local address |
| `__private void * __ockl_to_private(void *);` | Convert generic address to private address |
