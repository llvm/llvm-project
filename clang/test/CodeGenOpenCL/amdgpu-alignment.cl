// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char __attribute__((ext_vector_type(2))) char2;
typedef char __attribute__((ext_vector_type(3))) char3;
typedef char __attribute__((ext_vector_type(4))) char4;
typedef char __attribute__((ext_vector_type(8))) char8;
typedef char __attribute__((ext_vector_type(16))) char16;

typedef short __attribute__((ext_vector_type(2))) short2;
typedef short __attribute__((ext_vector_type(3))) short3;
typedef short __attribute__((ext_vector_type(4))) short4;
typedef short __attribute__((ext_vector_type(8))) short8;
typedef short __attribute__((ext_vector_type(16))) short16;

typedef int __attribute__((ext_vector_type(2))) int2;
typedef int __attribute__((ext_vector_type(3))) int3;
typedef int __attribute__((ext_vector_type(4))) int4;
typedef int __attribute__((ext_vector_type(8))) int8;
typedef int __attribute__((ext_vector_type(16))) int16;

typedef long __attribute__((ext_vector_type(2))) long2;
typedef long __attribute__((ext_vector_type(3))) long3;
typedef long __attribute__((ext_vector_type(4))) long4;
typedef long __attribute__((ext_vector_type(8))) long8;
typedef long __attribute__((ext_vector_type(16))) long16;

typedef half __attribute__((ext_vector_type(2))) half2;
typedef half __attribute__((ext_vector_type(3))) half3;
typedef half __attribute__((ext_vector_type(4))) half4;
typedef half __attribute__((ext_vector_type(8))) half8;
typedef half __attribute__((ext_vector_type(16))) half16;

typedef float __attribute__((ext_vector_type(2))) float2;
typedef float __attribute__((ext_vector_type(3))) float3;
typedef float __attribute__((ext_vector_type(4))) float4;
typedef float __attribute__((ext_vector_type(8))) float8;
typedef float __attribute__((ext_vector_type(16))) float16;

typedef double __attribute__((ext_vector_type(2))) double2;
typedef double __attribute__((ext_vector_type(3))) double3;
typedef double __attribute__((ext_vector_type(4))) double4;
typedef double __attribute__((ext_vector_type(8))) double8;
typedef double __attribute__((ext_vector_type(16))) double16;

// CHECK: @local_memory_alignment_global.lds_i8 = internal addrspace(3) global [4 x i8] undef, align 1
// CHECK: @local_memory_alignment_global.lds_v2i8 = internal addrspace(3) global [4 x <2 x i8>] undef, align 2
// CHECK: @local_memory_alignment_global.lds_v3i8 = internal addrspace(3) global [4 x <3 x i8>] undef, align 4
// CHECK: @local_memory_alignment_global.lds_v4i8 = internal addrspace(3) global [4 x <4 x i8>] undef, align 4
// CHECK: @local_memory_alignment_global.lds_v8i8 = internal addrspace(3) global [4 x <8 x i8>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v16i8 = internal addrspace(3) global [4 x <16 x i8>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_i16 = internal addrspace(3) global [4 x i16] undef, align 2
// CHECK: @local_memory_alignment_global.lds_v2i16 = internal addrspace(3) global [4 x <2 x i16>] undef, align 4
// CHECK: @local_memory_alignment_global.lds_v3i16 = internal addrspace(3) global [4 x <3 x i16>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v4i16 = internal addrspace(3) global [4 x <4 x i16>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v8i16 = internal addrspace(3) global [4 x <8 x i16>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v16i16 = internal addrspace(3) global [4 x <16 x i16>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_i32 = internal addrspace(3) global [4 x i32] undef, align 4
// CHECK: @local_memory_alignment_global.lds_v2i32 = internal addrspace(3) global [4 x <2 x i32>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v3i32 = internal addrspace(3) global [4 x <3 x i32>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v4i32 = internal addrspace(3) global [4 x <4 x i32>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v8i32 = internal addrspace(3) global [4 x <8 x i32>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_v16i32 = internal addrspace(3) global [4 x <16 x i32>] undef, align 64
// CHECK: @local_memory_alignment_global.lds_i64 = internal addrspace(3) global [4 x i64] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v2i64 = internal addrspace(3) global [4 x <2 x i64>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v3i64 = internal addrspace(3) global [4 x <3 x i64>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_v4i64 = internal addrspace(3) global [4 x <4 x i64>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_v8i64 = internal addrspace(3) global [4 x <8 x i64>] undef, align 64
// CHECK: @local_memory_alignment_global.lds_v16i64 = internal addrspace(3) global [4 x <16 x i64>] undef, align 128
// CHECK: @local_memory_alignment_global.lds_f16 = internal addrspace(3) global [4 x half] undef, align 2
// CHECK: @local_memory_alignment_global.lds_v2f16 = internal addrspace(3) global [4 x <2 x half>] undef, align 4
// CHECK: @local_memory_alignment_global.lds_v3f16 = internal addrspace(3) global [4 x <3 x half>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v4f16 = internal addrspace(3) global [4 x <4 x half>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v8f16 = internal addrspace(3) global [4 x <8 x half>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v16f16 = internal addrspace(3) global [4 x <16 x half>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_f32 = internal addrspace(3) global [4 x float] undef, align 4
// CHECK: @local_memory_alignment_global.lds_v2f32 = internal addrspace(3) global [4 x <2 x float>] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v3f32 = internal addrspace(3) global [4 x <3 x float>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v4f32 = internal addrspace(3) global [4 x <4 x float>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v8f32 = internal addrspace(3) global [4 x <8 x float>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_v16f32 = internal addrspace(3) global [4 x <16 x float>] undef, align 64
// CHECK: @local_memory_alignment_global.lds_f64 = internal addrspace(3) global [4 x double] undef, align 8
// CHECK: @local_memory_alignment_global.lds_v2f64 = internal addrspace(3) global [4 x <2 x double>] undef, align 16
// CHECK: @local_memory_alignment_global.lds_v3f64 = internal addrspace(3) global [4 x <3 x double>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_v4f64 = internal addrspace(3) global [4 x <4 x double>] undef, align 32
// CHECK: @local_memory_alignment_global.lds_v8f64 = internal addrspace(3) global [4 x <8 x double>] undef, align 64
// CHECK: @local_memory_alignment_global.lds_v16f64 = internal addrspace(3) global [4 x <16 x double>] undef, align 128


// CHECK-LABEL: @local_memory_alignment_global(
// CHECK: store volatile i8 0, i8 addrspace(3)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(3)* @local_memory_alignment_global.lds_i8, i32 0, i32 0), align 1
// CHECK: store volatile <2 x i8> zeroinitializer, <2 x i8> addrspace(3)* getelementptr inbounds ([4 x <2 x i8>], [4 x <2 x i8>] addrspace(3)* @local_memory_alignment_global.lds_v2i8, i32 0, i32 0), align 2
// CHECK: store volatile <4 x i8> <i8 0, i8 0, i8 0, i8 undef>, <4 x i8> addrspace(3)* bitcast ([4 x <3 x i8>] addrspace(3)* @local_memory_alignment_global.lds_v3i8 to <4 x i8> addrspace(3)*), align 4
// CHECK: store volatile <4 x i8> zeroinitializer, <4 x i8> addrspace(3)* getelementptr inbounds ([4 x <4 x i8>], [4 x <4 x i8>] addrspace(3)* @local_memory_alignment_global.lds_v4i8, i32 0, i32 0), align 4
// CHECK: store volatile <8 x i8> zeroinitializer, <8 x i8> addrspace(3)* getelementptr inbounds ([4 x <8 x i8>], [4 x <8 x i8>] addrspace(3)* @local_memory_alignment_global.lds_v8i8, i32 0, i32 0), align 8
// CHECK: store volatile <16 x i8> zeroinitializer, <16 x i8> addrspace(3)* getelementptr inbounds ([4 x <16 x i8>], [4 x <16 x i8>] addrspace(3)* @local_memory_alignment_global.lds_v16i8, i32 0, i32 0), align 16
// CHECK: store volatile i16 0, i16 addrspace(3)* getelementptr inbounds ([4 x i16], [4 x i16] addrspace(3)* @local_memory_alignment_global.lds_i16, i32 0, i32 0), align 2
// CHECK: store volatile <2 x i16> zeroinitializer, <2 x i16> addrspace(3)* getelementptr inbounds ([4 x <2 x i16>], [4 x <2 x i16>] addrspace(3)* @local_memory_alignment_global.lds_v2i16, i32 0, i32 0), align 4
// CHECK: store volatile <4 x i16> <i16 0, i16 0, i16 0, i16 undef>, <4 x i16> addrspace(3)* bitcast ([4 x <3 x i16>] addrspace(3)* @local_memory_alignment_global.lds_v3i16 to <4 x i16> addrspace(3)*), align 8
// CHECK: store volatile <4 x i16> zeroinitializer, <4 x i16> addrspace(3)* getelementptr inbounds ([4 x <4 x i16>], [4 x <4 x i16>] addrspace(3)* @local_memory_alignment_global.lds_v4i16, i32 0, i32 0), align 8
// CHECK: store volatile <8 x i16> zeroinitializer, <8 x i16> addrspace(3)* getelementptr inbounds ([4 x <8 x i16>], [4 x <8 x i16>] addrspace(3)* @local_memory_alignment_global.lds_v8i16, i32 0, i32 0), align 16
// CHECK: store volatile <16 x i16> zeroinitializer, <16 x i16> addrspace(3)* getelementptr inbounds ([4 x <16 x i16>], [4 x <16 x i16>] addrspace(3)* @local_memory_alignment_global.lds_v16i16, i32 0, i32 0), align 32
// CHECK: store volatile i32 0, i32 addrspace(3)* getelementptr inbounds ([4 x i32], [4 x i32] addrspace(3)* @local_memory_alignment_global.lds_i32, i32 0, i32 0), align 4
// CHECK: store volatile <2 x i32> zeroinitializer, <2 x i32> addrspace(3)* getelementptr inbounds ([4 x <2 x i32>], [4 x <2 x i32>] addrspace(3)* @local_memory_alignment_global.lds_v2i32, i32 0, i32 0), align 8
// CHECK: store volatile <4 x i32> <i32 0, i32 0, i32 0, i32 undef>, <4 x i32> addrspace(3)* bitcast ([4 x <3 x i32>] addrspace(3)* @local_memory_alignment_global.lds_v3i32 to <4 x i32> addrspace(3)*), align 16
// CHECK: store volatile <4 x i32> zeroinitializer, <4 x i32> addrspace(3)* getelementptr inbounds ([4 x <4 x i32>], [4 x <4 x i32>] addrspace(3)* @local_memory_alignment_global.lds_v4i32, i32 0, i32 0), align 16
// CHECK: store volatile <8 x i32> zeroinitializer, <8 x i32> addrspace(3)* getelementptr inbounds ([4 x <8 x i32>], [4 x <8 x i32>] addrspace(3)* @local_memory_alignment_global.lds_v8i32, i32 0, i32 0), align 32
// CHECK: store volatile <16 x i32> zeroinitializer, <16 x i32> addrspace(3)* getelementptr inbounds ([4 x <16 x i32>], [4 x <16 x i32>] addrspace(3)* @local_memory_alignment_global.lds_v16i32, i32 0, i32 0), align 64
// CHECK: store volatile i64 0, i64 addrspace(3)* getelementptr inbounds ([4 x i64], [4 x i64] addrspace(3)* @local_memory_alignment_global.lds_i64, i32 0, i32 0), align 8
// CHECK: store volatile <2 x i64> zeroinitializer, <2 x i64> addrspace(3)* getelementptr inbounds ([4 x <2 x i64>], [4 x <2 x i64>] addrspace(3)* @local_memory_alignment_global.lds_v2i64, i32 0, i32 0), align 16
// CHECK: store volatile <4 x i64> <i64 0, i64 0, i64 0, i64 undef>, <4 x i64> addrspace(3)* bitcast ([4 x <3 x i64>] addrspace(3)* @local_memory_alignment_global.lds_v3i64 to <4 x i64> addrspace(3)*), align 32
// CHECK: store volatile <4 x i64> zeroinitializer, <4 x i64> addrspace(3)* getelementptr inbounds ([4 x <4 x i64>], [4 x <4 x i64>] addrspace(3)* @local_memory_alignment_global.lds_v4i64, i32 0, i32 0), align 32
// CHECK: store volatile <8 x i64> zeroinitializer, <8 x i64> addrspace(3)* getelementptr inbounds ([4 x <8 x i64>], [4 x <8 x i64>] addrspace(3)* @local_memory_alignment_global.lds_v8i64, i32 0, i32 0), align 64
// CHECK: store volatile <16 x i64> zeroinitializer, <16 x i64> addrspace(3)* getelementptr inbounds ([4 x <16 x i64>], [4 x <16 x i64>] addrspace(3)* @local_memory_alignment_global.lds_v16i64, i32 0, i32 0), align 128
// CHECK: store volatile half 0xH0000, half addrspace(3)* getelementptr inbounds ([4 x half], [4 x half] addrspace(3)* @local_memory_alignment_global.lds_f16, i32 0, i32 0), align 2
// CHECK: store volatile <2 x half> zeroinitializer, <2 x half> addrspace(3)* getelementptr inbounds ([4 x <2 x half>], [4 x <2 x half>] addrspace(3)* @local_memory_alignment_global.lds_v2f16, i32 0, i32 0), align 4
// CHECK: store volatile <4 x half> <half 0xH0000, half 0xH0000, half 0xH0000, half undef>, <4 x half> addrspace(3)* bitcast ([4 x <3 x half>] addrspace(3)* @local_memory_alignment_global.lds_v3f16 to <4 x half> addrspace(3)*), align 8
// CHECK: store volatile <4 x half> zeroinitializer, <4 x half> addrspace(3)* getelementptr inbounds ([4 x <4 x half>], [4 x <4 x half>] addrspace(3)* @local_memory_alignment_global.lds_v4f16, i32 0, i32 0), align 8
// CHECK: store volatile <8 x half> zeroinitializer, <8 x half> addrspace(3)* getelementptr inbounds ([4 x <8 x half>], [4 x <8 x half>] addrspace(3)* @local_memory_alignment_global.lds_v8f16, i32 0, i32 0), align 16
// CHECK: store volatile <16 x half> zeroinitializer, <16 x half> addrspace(3)* getelementptr inbounds ([4 x <16 x half>], [4 x <16 x half>] addrspace(3)* @local_memory_alignment_global.lds_v16f16, i32 0, i32 0), align 32
// CHECK: store volatile float 0.000000e+00, float addrspace(3)* getelementptr inbounds ([4 x float], [4 x float] addrspace(3)* @local_memory_alignment_global.lds_f32, i32 0, i32 0), align 4
// CHECK: store volatile <2 x float> zeroinitializer, <2 x float> addrspace(3)* getelementptr inbounds ([4 x <2 x float>], [4 x <2 x float>] addrspace(3)* @local_memory_alignment_global.lds_v2f32, i32 0, i32 0), align 8
// CHECK: store volatile <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float undef>, <4 x float> addrspace(3)* bitcast ([4 x <3 x float>] addrspace(3)* @local_memory_alignment_global.lds_v3f32 to <4 x float> addrspace(3)*), align 16
// CHECK: store volatile <4 x float> zeroinitializer, <4 x float> addrspace(3)* getelementptr inbounds ([4 x <4 x float>], [4 x <4 x float>] addrspace(3)* @local_memory_alignment_global.lds_v4f32, i32 0, i32 0), align 16
// CHECK: store volatile <8 x float> zeroinitializer, <8 x float> addrspace(3)* getelementptr inbounds ([4 x <8 x float>], [4 x <8 x float>] addrspace(3)* @local_memory_alignment_global.lds_v8f32, i32 0, i32 0), align 32
// CHECK: store volatile <16 x float> zeroinitializer, <16 x float> addrspace(3)* getelementptr inbounds ([4 x <16 x float>], [4 x <16 x float>] addrspace(3)* @local_memory_alignment_global.lds_v16f32, i32 0, i32 0), align 64
// CHECK: store volatile double 0.000000e+00, double addrspace(3)* getelementptr inbounds ([4 x double], [4 x double] addrspace(3)* @local_memory_alignment_global.lds_f64, i32 0, i32 0), align 8
// CHECK: store volatile <2 x double> zeroinitializer, <2 x double> addrspace(3)* getelementptr inbounds ([4 x <2 x double>], [4 x <2 x double>] addrspace(3)* @local_memory_alignment_global.lds_v2f64, i32 0, i32 0), align 16
// CHECK: store volatile <4 x double> <double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double undef>, <4 x double> addrspace(3)* bitcast ([4 x <3 x double>] addrspace(3)* @local_memory_alignment_global.lds_v3f64 to <4 x double> addrspace(3)*), align 32
// CHECK: store volatile <4 x double> zeroinitializer, <4 x double> addrspace(3)* getelementptr inbounds ([4 x <4 x double>], [4 x <4 x double>] addrspace(3)* @local_memory_alignment_global.lds_v4f64, i32 0, i32 0), align 32
// CHECK: store volatile <8 x double> zeroinitializer, <8 x double> addrspace(3)* getelementptr inbounds ([4 x <8 x double>], [4 x <8 x double>] addrspace(3)* @local_memory_alignment_global.lds_v8f64, i32 0, i32 0), align 64
// CHECK: store volatile <16 x double> zeroinitializer, <16 x double> addrspace(3)* getelementptr inbounds ([4 x <16 x double>], [4 x <16 x double>] addrspace(3)* @local_memory_alignment_global.lds_v16f64, i32 0, i32 0), align 128
kernel void local_memory_alignment_global()
{
  volatile local char lds_i8[4];
  volatile local char2 lds_v2i8[4];
  volatile local char3 lds_v3i8[4];
  volatile local char4 lds_v4i8[4];
  volatile local char8 lds_v8i8[4];
  volatile local char16 lds_v16i8[4];

  volatile local short lds_i16[4];
  volatile local short2 lds_v2i16[4];
  volatile local short3 lds_v3i16[4];
  volatile local short4 lds_v4i16[4];
  volatile local short8 lds_v8i16[4];
  volatile local short16 lds_v16i16[4];

  volatile local int lds_i32[4];
  volatile local int2 lds_v2i32[4];
  volatile local int3 lds_v3i32[4];
  volatile local int4 lds_v4i32[4];
  volatile local int8 lds_v8i32[4];
  volatile local int16 lds_v16i32[4];

  volatile local long lds_i64[4];
  volatile local long2 lds_v2i64[4];
  volatile local long3 lds_v3i64[4];
  volatile local long4 lds_v4i64[4];
  volatile local long8 lds_v8i64[4];
  volatile local long16 lds_v16i64[4];

  volatile local half lds_f16[4];
  volatile local half2 lds_v2f16[4];
  volatile local half3 lds_v3f16[4];
  volatile local half4 lds_v4f16[4];
  volatile local half8 lds_v8f16[4];
  volatile local half16 lds_v16f16[4];

  volatile local float lds_f32[4];
  volatile local float2 lds_v2f32[4];
  volatile local float3 lds_v3f32[4];
  volatile local float4 lds_v4f32[4];
  volatile local float8 lds_v8f32[4];
  volatile local float16 lds_v16f32[4];

  volatile local double lds_f64[4];
  volatile local double2 lds_v2f64[4];
  volatile local double3 lds_v3f64[4];
  volatile local double4 lds_v4f64[4];
  volatile local double8 lds_v8f64[4];
  volatile local double16 lds_v16f64[4];

  *lds_i8 = 0;
  *lds_v2i8 = 0;
  *lds_v3i8 = 0;
  *lds_v4i8 = 0;
  *lds_v8i8 = 0;
  *lds_v16i8 = 0;

  *lds_i16 = 0;
  *lds_v2i16 = 0;
  *lds_v3i16 = 0;
  *lds_v4i16 = 0;
  *lds_v8i16 = 0;
  *lds_v16i16 = 0;

  *lds_i32 = 0;
  *lds_v2i32 = 0;
  *lds_v3i32 = 0;
  *lds_v4i32 = 0;
  *lds_v8i32 = 0;
  *lds_v16i32 = 0;

  *lds_i64 = 0;
  *lds_v2i64 = 0;
  *lds_v3i64 = 0;
  *lds_v4i64 = 0;
  *lds_v8i64 = 0;
  *lds_v16i64 = 0;

  *lds_f16 = 0;
  *lds_v2f16 = 0;
  *lds_v3f16 = 0;
  *lds_v4f16 = 0;
  *lds_v8f16 = 0;
  *lds_v16f16 = 0;

  *lds_f32 = 0;
  *lds_v2f32 = 0;
  *lds_v3f32 = 0;
  *lds_v4f32 = 0;
  *lds_v8f32 = 0;
  *lds_v16f32 = 0;

  *lds_f64 = 0;
  *lds_v2f64 = 0;
  *lds_v3f64 = 0;
  *lds_v4f64 = 0;
  *lds_v8f64 = 0;
  *lds_v16f64 = 0;
}

kernel void local_memory_alignment_arg(
  volatile local char* lds_i8,
  volatile local char2* lds_v2i8,
  volatile local char3* lds_v3i8,
  volatile local char4* lds_v4i8,
  volatile local char8* lds_v8i8,
  volatile local char16* lds_v16i8,

  volatile local short* lds_i16,
  volatile local short2* lds_v2i16,
  volatile local short3* lds_v3i16,
  volatile local short4* lds_v4i16,
  volatile local short8* lds_v8i16,
  volatile local short16* lds_v16i16,

  volatile local int* lds_i32,
  volatile local int2* lds_v2i32,
  volatile local int3* lds_v3i32,
  volatile local int4* lds_v4i32,
  volatile local int8* lds_v8i32,
  volatile local int16* lds_v16i32,

  volatile local long* lds_i64,
  volatile local long2* lds_v2i64,
  volatile local long3* lds_v3i64,
  volatile local long4* lds_v4i64,
  volatile local long8* lds_v8i64,
  volatile local long16* lds_v16i64,

  volatile local half* lds_f16,
  volatile local half2* lds_v2f16,
  volatile local half3* lds_v3f16,
  volatile local half4* lds_v4f16,
  volatile local half8* lds_v8f16,
  volatile local half16* lds_v16f16,

  volatile local float* lds_f32,
  volatile local float2* lds_v2f32,
  volatile local float3* lds_v3f32,
  volatile local float4* lds_v4f32,
  volatile local float8* lds_v8f32,
  volatile local float16* lds_v16f32,

  volatile local double* lds_f64,
  volatile local double2* lds_v2f64,
  volatile local double3* lds_v3f64,
  volatile local double4* lds_v4f64,
  volatile local double8* lds_v8f64,
  volatile local double16* lds_v16f64)
{
  *lds_i8 = 0;
  *lds_v2i8 = 0;
  *lds_v3i8 = 0;
  *lds_v4i8 = 0;
  *lds_v8i8 = 0;
  *lds_v16i8 = 0;

  *lds_i16 = 0;
  *lds_v2i16 = 0;
  *lds_v3i16 = 0;
  *lds_v4i16 = 0;
  *lds_v8i16 = 0;
  *lds_v16i16 = 0;

  *lds_i32 = 0;
  *lds_v2i32 = 0;
  *lds_v3i32 = 0;
  *lds_v4i32 = 0;
  *lds_v8i32 = 0;
  *lds_v16i32 = 0;

  *lds_i64 = 0;
  *lds_v2i64 = 0;
  *lds_v3i64 = 0;
  *lds_v4i64 = 0;
  *lds_v8i64 = 0;
  *lds_v16i64 = 0;

  *lds_f16 = 0;
  *lds_v2f16 = 0;
  *lds_v3f16 = 0;
  *lds_v4f16 = 0;
  *lds_v8f16 = 0;
  *lds_v16f16 = 0;

  *lds_f32 = 0;
  *lds_v2f32 = 0;
  *lds_v3f32 = 0;
  *lds_v4f32 = 0;
  *lds_v8f32 = 0;
  *lds_v16f32 = 0;

  *lds_f64 = 0;
  *lds_v2f64 = 0;
  *lds_v3f64 = 0;
  *lds_v4f64 = 0;
  *lds_v8f64 = 0;
  *lds_v16f64 = 0;
}

// CHECK-LABEL: @private_memory_alignment_alloca(
// CHECK: %private_i8 = alloca [4 x i8], align 1
// CHECK: %private_v2i8 = alloca [4 x <2 x i8>], align 2
// CHECK: %private_v3i8 = alloca [4 x <3 x i8>], align 4
// CHECK: %private_v4i8 = alloca [4 x <4 x i8>], align 4
// CHECK: %private_v8i8 = alloca [4 x <8 x i8>], align 8
// CHECK: %private_v16i8 = alloca [4 x <16 x i8>], align 16
// CHECK: %private_i16 = alloca [4 x i16], align 2
// CHECK: %private_v2i16 = alloca [4 x <2 x i16>], align 4
// CHECK: %private_v3i16 = alloca [4 x <3 x i16>], align 8
// CHECK: %private_v4i16 = alloca [4 x <4 x i16>], align 8
// CHECK: %private_v8i16 = alloca [4 x <8 x i16>], align 16
// CHECK: %private_v16i16 = alloca [4 x <16 x i16>], align 32
// CHECK: %private_i32 = alloca [4 x i32], align 4
// CHECK: %private_v2i32 = alloca [4 x <2 x i32>], align 8
// CHECK: %private_v3i32 = alloca [4 x <3 x i32>], align 16
// CHECK: %private_v4i32 = alloca [4 x <4 x i32>], align 16
// CHECK: %private_v8i32 = alloca [4 x <8 x i32>], align 32
// CHECK: %private_v16i32 = alloca [4 x <16 x i32>], align 64
// CHECK: %private_i64 = alloca [4 x i64], align 8
// CHECK: %private_v2i64 = alloca [4 x <2 x i64>], align 16
// CHECK: %private_v3i64 = alloca [4 x <3 x i64>], align 32
// CHECK: %private_v4i64 = alloca [4 x <4 x i64>], align 32
// CHECK: %private_v8i64 = alloca [4 x <8 x i64>], align 64
// CHECK: %private_v16i64 = alloca [4 x <16 x i64>], align 128
// CHECK: %private_f16 = alloca [4 x half], align 2
// CHECK: %private_v2f16 = alloca [4 x <2 x half>], align 4
// CHECK: %private_v3f16 = alloca [4 x <3 x half>], align 8
// CHECK: %private_v4f16 = alloca [4 x <4 x half>], align 8
// CHECK: %private_v8f16 = alloca [4 x <8 x half>], align 16
// CHECK: %private_v16f16 = alloca [4 x <16 x half>], align 32
// CHECK: %private_f32 = alloca [4 x float], align 4
// CHECK: %private_v2f32 = alloca [4 x <2 x float>], align 8
// CHECK: %private_v3f32 = alloca [4 x <3 x float>], align 16
// CHECK: %private_v4f32 = alloca [4 x <4 x float>], align 16
// CHECK: %private_v8f32 = alloca [4 x <8 x float>], align 32
// CHECK: %private_v16f32 = alloca [4 x <16 x float>], align 64
// CHECK: %private_f64 = alloca [4 x double], align 8
// CHECK: %private_v2f64 = alloca [4 x <2 x double>], align 16
// CHECK: %private_v3f64 = alloca [4 x <3 x double>], align 32
// CHECK: %private_v4f64 = alloca [4 x <4 x double>], align 32
// CHECK: %private_v8f64 = alloca [4 x <8 x double>], align 64
// CHECK: %private_v16f64 = alloca [4 x <16 x double>], align 128

// CHECK: store volatile i8 0, i8* %arraydecay, align 1
// CHECK: store volatile <2 x i8> zeroinitializer, <2 x i8>* %arraydecay{{[0-9]+}}, align 2
// CHECK: store volatile <4 x i8> <i8 0, i8 0, i8 0, i8 undef>, <4 x i8>* %storetmp, align 4
// CHECK: store volatile <4 x i8> zeroinitializer, <4 x i8>* %arraydecay{{[0-9]+}}, align 4
// CHECK: store volatile <8 x i8> zeroinitializer, <8 x i8>* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <16 x i8> zeroinitializer, <16 x i8>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile i16 0, i16* %arraydecay{{[0-9]+}}, align 2
// CHECK: store volatile <2 x i16> zeroinitializer, <2 x i16>* %arraydecay{{[0-9]+}}, align 4
// CHECK: store volatile <4 x i16> <i16 0, i16 0, i16 0, i16 undef>, <4 x i16>* %storetmp{{[0-9]+}}, align 8
// CHECK: store volatile <4 x i16> zeroinitializer, <4 x i16>* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <8 x i16> zeroinitializer, <8 x i16>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile <16 x i16> zeroinitializer, <16 x i16>* %arraydecay{{[0-9]+}}, align 32
// CHECK: store volatile i32 0, i32* %arraydecay{{[0-9]+}}, align 4
// CHECK: store volatile <2 x i32> zeroinitializer, <2 x i32>* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <4 x i32> <i32 0, i32 0, i32 0, i32 undef>, <4 x i32>* %storetmp16, align 16
// CHECK: store volatile <4 x i32> zeroinitializer, <4 x i32>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile <8 x i32> zeroinitializer, <8 x i32>* %arraydecay{{[0-9]+}}, align 32
// CHECK: store volatile <16 x i32> zeroinitializer, <16 x i32>* %arraydecay{{[0-9]+}}, align 64
// CHECK: store volatile i64 0, i64* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <2 x i64> zeroinitializer, <2 x i64>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile <4 x i64> <i64 0, i64 0, i64 0, i64 undef>, <4 x i64>* %storetmp23, align 32
// CHECK: store volatile <4 x i64> zeroinitializer, <4 x i64>* %arraydecay{{[0-9]+}}, align 32
// CHECK: store volatile <8 x i64> zeroinitializer, <8 x i64>* %arraydecay{{[0-9]+}}, align 64
// CHECK: store volatile <16 x i64> zeroinitializer, <16 x i64>* %arraydecay{{[0-9]+}}, align 128
// CHECK: store volatile half 0xH0000, half* %arraydecay{{[0-9]+}}, align 2
// CHECK: store volatile <2 x half> zeroinitializer, <2 x half>* %arraydecay{{[0-9]+}}, align 4
// CHECK: store volatile <4 x half> <half 0xH0000, half 0xH0000, half 0xH0000, half undef>, <4 x half>* %storetmp{{[0-9]+}}, align 8
// CHECK: store volatile <4 x half> zeroinitializer, <4 x half>* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <8 x half> zeroinitializer, <8 x half>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile <16 x half> zeroinitializer, <16 x half>* %arraydecay{{[0-9]+}}, align 32
// CHECK: store volatile float 0.000000e+00, float* %arraydecay34, align 4
// CHECK: store volatile <2 x float> zeroinitializer, <2 x float>* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float undef>, <4 x float>* %storetmp{{[0-9]+}}, align 16
// CHECK: store volatile <4 x float> zeroinitializer, <4 x float>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile <8 x float> zeroinitializer, <8 x float>* %arraydecay{{[0-9]+}}, align 32
// CHECK: store volatile <16 x float> zeroinitializer, <16 x float>* %arraydecay{{[0-9]+}}, align 64
// CHECK: store volatile double 0.000000e+00, double* %arraydecay{{[0-9]+}}, align 8
// CHECK: store volatile <2 x double> zeroinitializer, <2 x double>* %arraydecay{{[0-9]+}}, align 16
// CHECK: store volatile <4 x double> <double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double undef>, <4 x double>* %storetmp{{[0-9]+}}, align 32
// CHECK: store volatile <4 x double> zeroinitializer, <4 x double>* %arraydecay{{[0-9]+}}, align 32
// CHECK: store volatile <8 x double> zeroinitializer, <8 x double>* %arraydecay{{[0-9]+}}, align 64
// CHECK: store volatile <16 x double> zeroinitializer, <16 x double>* %arraydecay{{[0-9]+}}, align 128
kernel void private_memory_alignment_alloca()
{
  volatile private char private_i8[4];
  volatile private char2 private_v2i8[4];
  volatile private char3 private_v3i8[4];
  volatile private char4 private_v4i8[4];
  volatile private char8 private_v8i8[4];
  volatile private char16 private_v16i8[4];

  volatile private short private_i16[4];
  volatile private short2 private_v2i16[4];
  volatile private short3 private_v3i16[4];
  volatile private short4 private_v4i16[4];
  volatile private short8 private_v8i16[4];
  volatile private short16 private_v16i16[4];

  volatile private int private_i32[4];
  volatile private int2 private_v2i32[4];
  volatile private int3 private_v3i32[4];
  volatile private int4 private_v4i32[4];
  volatile private int8 private_v8i32[4];
  volatile private int16 private_v16i32[4];

  volatile private long private_i64[4];
  volatile private long2 private_v2i64[4];
  volatile private long3 private_v3i64[4];
  volatile private long4 private_v4i64[4];
  volatile private long8 private_v8i64[4];
  volatile private long16 private_v16i64[4];

  volatile private half private_f16[4];
  volatile private half2 private_v2f16[4];
  volatile private half3 private_v3f16[4];
  volatile private half4 private_v4f16[4];
  volatile private half8 private_v8f16[4];
  volatile private half16 private_v16f16[4];

  volatile private float private_f32[4];
  volatile private float2 private_v2f32[4];
  volatile private float3 private_v3f32[4];
  volatile private float4 private_v4f32[4];
  volatile private float8 private_v8f32[4];
  volatile private float16 private_v16f32[4];

  volatile private double private_f64[4];
  volatile private double2 private_v2f64[4];
  volatile private double3 private_v3f64[4];
  volatile private double4 private_v4f64[4];
  volatile private double8 private_v8f64[4];
  volatile private double16 private_v16f64[4];

  *private_i8 = 0;
  *private_v2i8 = 0;
  *private_v3i8 = 0;
  *private_v4i8 = 0;
  *private_v8i8 = 0;
  *private_v16i8 = 0;

  *private_i16 = 0;
  *private_v2i16 = 0;
  *private_v3i16 = 0;
  *private_v4i16 = 0;
  *private_v8i16 = 0;
  *private_v16i16 = 0;

  *private_i32 = 0;
  *private_v2i32 = 0;
  *private_v3i32 = 0;
  *private_v4i32 = 0;
  *private_v8i32 = 0;
  *private_v16i32 = 0;

  *private_i64 = 0;
  *private_v2i64 = 0;
  *private_v3i64 = 0;
  *private_v4i64 = 0;
  *private_v8i64 = 0;
  *private_v16i64 = 0;

  *private_f16 = 0;
  *private_v2f16 = 0;
  *private_v3f16 = 0;
  *private_v4f16 = 0;
  *private_v8f16 = 0;
  *private_v16f16 = 0;

  *private_f32 = 0;
  *private_v2f32 = 0;
  *private_v3f32 = 0;
  *private_v4f32 = 0;
  *private_v8f32 = 0;
  *private_v16f32 = 0;

  *private_f64 = 0;
  *private_v2f64 = 0;
  *private_v3f64 = 0;
  *private_v4f64 = 0;
  *private_v8f64 = 0;
  *private_v16f64 = 0;
}
