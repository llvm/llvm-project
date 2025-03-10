// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - -DNAMESPACED| FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - -DSPIRV| FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - -DSPIRV -DNAMESPACED| FileCheck %s

#ifdef NAMESPACED
#define TYPE_DECL(T)  hlsl::T T##_Val
#else
#define TYPE_DECL(T)  T T##_Val
#endif

// Until MicrosoftCXXABI supports mangling matrices,
// these have to be local variables for DXIL.
#ifndef SPIRV
void f() {
#endif

// built-in matrix types:

// Capture target-specific details.
//CHECK: [[PFX:[%@]]]int16_t1x1_Val = [[STR:(alloca|global)]] [1 x i16][[ZI:( zeroinitializer)?]], align 2
//CHECK: [[PFX]]int16_t1x2_Val = [[STR]] [2 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t1x3_Val = [[STR]] [3 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t1x4_Val = [[STR]] [4 x i16][[ZI]], align 2
TYPE_DECL( int16_t1x1 );
TYPE_DECL( int16_t1x2 );
TYPE_DECL( int16_t1x3 );
TYPE_DECL( int16_t1x4 );

//CHECK: [[PFX]]int16_t2x1_Val = [[STR]] [2 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t2x2_Val = [[STR]] [4 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t2x3_Val = [[STR]] [6 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t2x4_Val = [[STR]] [8 x i16][[ZI]], align 2
TYPE_DECL( int16_t2x1 );
TYPE_DECL( int16_t2x2 );
TYPE_DECL( int16_t2x3 );
TYPE_DECL( int16_t2x4 );

//CHECK: [[PFX]]int16_t3x1_Val = [[STR]] [3 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t3x2_Val = [[STR]] [6 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t3x3_Val = [[STR]] [9 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t3x4_Val = [[STR]] [12 x i16][[ZI]], align 2
TYPE_DECL( int16_t3x1 );
TYPE_DECL( int16_t3x2 );
TYPE_DECL( int16_t3x3 );
TYPE_DECL( int16_t3x4 );

//CHECK: [[PFX]]int16_t4x1_Val = [[STR]] [4 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t4x2_Val = [[STR]] [8 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t4x3_Val = [[STR]] [12 x i16][[ZI]], align 2
//CHECK: [[PFX]]int16_t4x4_Val = [[STR]] [16 x i16][[ZI]], align 2
TYPE_DECL( int16_t4x1 );
TYPE_DECL( int16_t4x2 );
TYPE_DECL( int16_t4x3 );
TYPE_DECL( int16_t4x4 );

//CHECK: [[PFX]]uint16_t1x1_Val = [[STR]] [1 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t1x2_Val = [[STR]] [2 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t1x3_Val = [[STR]] [3 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t1x4_Val = [[STR]] [4 x i16][[ZI]], align 2
TYPE_DECL( uint16_t1x1 );
TYPE_DECL( uint16_t1x2 );
TYPE_DECL( uint16_t1x3 );
TYPE_DECL( uint16_t1x4 );

//CHECK: [[PFX]]uint16_t2x1_Val = [[STR]] [2 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t2x2_Val = [[STR]] [4 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t2x3_Val = [[STR]] [6 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t2x4_Val = [[STR]] [8 x i16][[ZI]], align 2
TYPE_DECL( uint16_t2x1 );
TYPE_DECL( uint16_t2x2 );
TYPE_DECL( uint16_t2x3 );
TYPE_DECL( uint16_t2x4 );

//CHECK: [[PFX]]uint16_t3x1_Val = [[STR]] [3 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t3x2_Val = [[STR]] [6 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t3x3_Val = [[STR]] [9 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t3x4_Val = [[STR]] [12 x i16][[ZI]], align 2
TYPE_DECL( uint16_t3x1 );
TYPE_DECL( uint16_t3x2 );
TYPE_DECL( uint16_t3x3 );
TYPE_DECL( uint16_t3x4 );

//CHECK: [[PFX]]uint16_t4x1_Val = [[STR]] [4 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t4x2_Val = [[STR]] [8 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t4x3_Val = [[STR]] [12 x i16][[ZI]], align 2
//CHECK: [[PFX]]uint16_t4x4_Val = [[STR]] [16 x i16][[ZI]], align 2
TYPE_DECL( uint16_t4x1 );
TYPE_DECL( uint16_t4x2 );
TYPE_DECL( uint16_t4x3 );
TYPE_DECL( uint16_t4x4 );

//CHECK: [[PFX]]int1x1_Val = [[STR]] [1 x i32][[ZI]], align 4
//CHECK: [[PFX]]int1x2_Val = [[STR]] [2 x i32][[ZI]], align 4
//CHECK: [[PFX]]int1x3_Val = [[STR]] [3 x i32][[ZI]], align 4
//CHECK: [[PFX]]int1x4_Val = [[STR]] [4 x i32][[ZI]], align 4
TYPE_DECL( int1x1 );
TYPE_DECL( int1x2 );
TYPE_DECL( int1x3 );
TYPE_DECL( int1x4 );

//CHECK: [[PFX]]int2x1_Val = [[STR]] [2 x i32][[ZI]], align 4
//CHECK: [[PFX]]int2x2_Val = [[STR]] [4 x i32][[ZI]], align 4
//CHECK: [[PFX]]int2x3_Val = [[STR]] [6 x i32][[ZI]], align 4
//CHECK: [[PFX]]int2x4_Val = [[STR]] [8 x i32][[ZI]], align 4
TYPE_DECL( int2x1 );
TYPE_DECL( int2x2 );
TYPE_DECL( int2x3 );
TYPE_DECL( int2x4 );

//CHECK: [[PFX]]int3x1_Val = [[STR]] [3 x i32][[ZI]], align 4
//CHECK: [[PFX]]int3x2_Val = [[STR]] [6 x i32][[ZI]], align 4
//CHECK: [[PFX]]int3x3_Val = [[STR]] [9 x i32][[ZI]], align 4
//CHECK: [[PFX]]int3x4_Val = [[STR]] [12 x i32][[ZI]], align 4
TYPE_DECL( int3x1 );
TYPE_DECL( int3x2 );
TYPE_DECL( int3x3 );
TYPE_DECL( int3x4 );

//CHECK: [[PFX]]int4x1_Val = [[STR]] [4 x i32][[ZI]], align 4
//CHECK: [[PFX]]int4x2_Val = [[STR]] [8 x i32][[ZI]], align 4
//CHECK: [[PFX]]int4x3_Val = [[STR]] [12 x i32][[ZI]], align 4
//CHECK: [[PFX]]int4x4_Val = [[STR]] [16 x i32][[ZI]], align 4
TYPE_DECL( int4x1 );
TYPE_DECL( int4x2 );
TYPE_DECL( int4x3 );
TYPE_DECL( int4x4 );

//CHECK: [[PFX]]uint1x1_Val = [[STR]] [1 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint1x2_Val = [[STR]] [2 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint1x3_Val = [[STR]] [3 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint1x4_Val = [[STR]] [4 x i32][[ZI]], align 4
TYPE_DECL( uint1x1 );
TYPE_DECL( uint1x2 );
TYPE_DECL( uint1x3 );
TYPE_DECL( uint1x4 );

//CHECK: [[PFX]]uint2x1_Val = [[STR]] [2 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint2x2_Val = [[STR]] [4 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint2x3_Val = [[STR]] [6 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint2x4_Val = [[STR]] [8 x i32][[ZI]], align 4
TYPE_DECL( uint2x1 );
TYPE_DECL( uint2x2 );
TYPE_DECL( uint2x3 );
TYPE_DECL( uint2x4 );

//CHECK: [[PFX]]uint3x1_Val = [[STR]] [3 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint3x2_Val = [[STR]] [6 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint3x3_Val = [[STR]] [9 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint3x4_Val = [[STR]] [12 x i32][[ZI]], align 4
TYPE_DECL( uint3x1 );
TYPE_DECL( uint3x2 );
TYPE_DECL( uint3x3 );
TYPE_DECL( uint3x4 );

//CHECK: [[PFX]]uint4x1_Val = [[STR]] [4 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint4x2_Val = [[STR]] [8 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint4x3_Val = [[STR]] [12 x i32][[ZI]], align 4
//CHECK: [[PFX]]uint4x4_Val = [[STR]] [16 x i32][[ZI]], align 4
TYPE_DECL( uint4x1 );
TYPE_DECL( uint4x2 );
TYPE_DECL( uint4x3 );
TYPE_DECL( uint4x4 );

//CHECK: [[PFX]]int64_t1x1_Val = [[STR]] [1 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t1x2_Val = [[STR]] [2 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t1x3_Val = [[STR]] [3 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t1x4_Val = [[STR]] [4 x i64][[ZI]], align 8
TYPE_DECL( int64_t1x1 );
TYPE_DECL( int64_t1x2 );
TYPE_DECL( int64_t1x3 );
TYPE_DECL( int64_t1x4 );

//CHECK: [[PFX]]int64_t2x1_Val = [[STR]] [2 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t2x2_Val = [[STR]] [4 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t2x3_Val = [[STR]] [6 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t2x4_Val = [[STR]] [8 x i64][[ZI]], align 8
TYPE_DECL( int64_t2x1 );
TYPE_DECL( int64_t2x2 );
TYPE_DECL( int64_t2x3 );
TYPE_DECL( int64_t2x4 );

//CHECK: [[PFX]]int64_t3x1_Val = [[STR]] [3 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t3x2_Val = [[STR]] [6 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t3x3_Val = [[STR]] [9 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t3x4_Val = [[STR]] [12 x i64][[ZI]], align 8
TYPE_DECL( int64_t3x1 );
TYPE_DECL( int64_t3x2 );
TYPE_DECL( int64_t3x3 );
TYPE_DECL( int64_t3x4 );

//CHECK: [[PFX]]int64_t4x1_Val = [[STR]] [4 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t4x2_Val = [[STR]] [8 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t4x3_Val = [[STR]] [12 x i64][[ZI]], align 8
//CHECK: [[PFX]]int64_t4x4_Val = [[STR]] [16 x i64][[ZI]], align 8
TYPE_DECL( int64_t4x1 );
TYPE_DECL( int64_t4x2 );
TYPE_DECL( int64_t4x3 );
TYPE_DECL( int64_t4x4 );

//CHECK: [[PFX]]uint64_t1x1_Val = [[STR]] [1 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t1x2_Val = [[STR]] [2 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t1x3_Val = [[STR]] [3 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t1x4_Val = [[STR]] [4 x i64][[ZI]], align 8
TYPE_DECL( uint64_t1x1 );
TYPE_DECL( uint64_t1x2 );
TYPE_DECL( uint64_t1x3 );
TYPE_DECL( uint64_t1x4 );

//CHECK: [[PFX]]uint64_t2x1_Val = [[STR]] [2 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t2x2_Val = [[STR]] [4 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t2x3_Val = [[STR]] [6 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t2x4_Val = [[STR]] [8 x i64][[ZI]], align 8
TYPE_DECL( uint64_t2x1 );
TYPE_DECL( uint64_t2x2 );
TYPE_DECL( uint64_t2x3 );
TYPE_DECL( uint64_t2x4 );

//CHECK: [[PFX]]uint64_t3x1_Val = [[STR]] [3 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t3x2_Val = [[STR]] [6 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t3x3_Val = [[STR]] [9 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t3x4_Val = [[STR]] [12 x i64][[ZI]], align 8
TYPE_DECL( uint64_t3x1 );
TYPE_DECL( uint64_t3x2 );
TYPE_DECL( uint64_t3x3 );
TYPE_DECL( uint64_t3x4 );

//CHECK: [[PFX]]uint64_t4x1_Val = [[STR]] [4 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t4x2_Val = [[STR]] [8 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t4x3_Val = [[STR]] [12 x i64][[ZI]], align 8
//CHECK: [[PFX]]uint64_t4x4_Val = [[STR]] [16 x i64][[ZI]], align 8
TYPE_DECL( uint64_t4x1 );
TYPE_DECL( uint64_t4x2 );
TYPE_DECL( uint64_t4x3 );
TYPE_DECL( uint64_t4x4 );


//CHECK: [[PFX]]half1x1_Val = [[STR]] [1 x half][[ZI]], align 2
//CHECK: [[PFX]]half1x2_Val = [[STR]] [2 x half][[ZI]], align 2
//CHECK: [[PFX]]half1x3_Val = [[STR]] [3 x half][[ZI]], align 2
//CHECK: [[PFX]]half1x4_Val = [[STR]] [4 x half][[ZI]], align 2
TYPE_DECL( half1x1 );
TYPE_DECL( half1x2 );
TYPE_DECL( half1x3 );
TYPE_DECL( half1x4 );

//CHECK: [[PFX]]half2x1_Val = [[STR]] [2 x half][[ZI]], align 2
//CHECK: [[PFX]]half2x2_Val = [[STR]] [4 x half][[ZI]], align 2
//CHECK: [[PFX]]half2x3_Val = [[STR]] [6 x half][[ZI]], align 2
//CHECK: [[PFX]]half2x4_Val = [[STR]] [8 x half][[ZI]], align 2
TYPE_DECL( half2x1 );
TYPE_DECL( half2x2 );
TYPE_DECL( half2x3 );
TYPE_DECL( half2x4 );

//CHECK: [[PFX]]half3x1_Val = [[STR]] [3 x half][[ZI]], align 2
//CHECK: [[PFX]]half3x2_Val = [[STR]] [6 x half][[ZI]], align 2
//CHECK: [[PFX]]half3x3_Val = [[STR]] [9 x half][[ZI]], align 2
//CHECK: [[PFX]]half3x4_Val = [[STR]] [12 x half][[ZI]], align 2
TYPE_DECL( half3x1 );
TYPE_DECL( half3x2 );
TYPE_DECL( half3x3 );
TYPE_DECL( half3x4 );

//CHECK: [[PFX]]half4x1_Val = [[STR]] [4 x half][[ZI]], align 2
//CHECK: [[PFX]]half4x2_Val = [[STR]] [8 x half][[ZI]], align 2
//CHECK: [[PFX]]half4x3_Val = [[STR]] [12 x half][[ZI]], align 2
//CHECK: [[PFX]]half4x4_Val = [[STR]] [16 x half][[ZI]], align 2
TYPE_DECL( half4x1 );
TYPE_DECL( half4x2 );
TYPE_DECL( half4x3 );
TYPE_DECL( half4x4 );

//CHECK: [[PFX]]float1x1_Val = [[STR]] [1 x float][[ZI]], align 4
//CHECK: [[PFX]]float1x2_Val = [[STR]] [2 x float][[ZI]], align 4
//CHECK: [[PFX]]float1x3_Val = [[STR]] [3 x float][[ZI]], align 4
//CHECK: [[PFX]]float1x4_Val = [[STR]] [4 x float][[ZI]], align 4
TYPE_DECL( float1x1 );
TYPE_DECL( float1x2 );
TYPE_DECL( float1x3 );
TYPE_DECL( float1x4 );

//CHECK: [[PFX]]float2x1_Val = [[STR]] [2 x float][[ZI]], align 4
//CHECK: [[PFX]]float2x2_Val = [[STR]] [4 x float][[ZI]], align 4
//CHECK: [[PFX]]float2x3_Val = [[STR]] [6 x float][[ZI]], align 4
//CHECK: [[PFX]]float2x4_Val = [[STR]] [8 x float][[ZI]], align 4
TYPE_DECL( float2x1 );
TYPE_DECL( float2x2 );
TYPE_DECL( float2x3 );
TYPE_DECL( float2x4 );

//CHECK: [[PFX]]float3x1_Val = [[STR]] [3 x float][[ZI]], align 4
//CHECK: [[PFX]]float3x2_Val = [[STR]] [6 x float][[ZI]], align 4
//CHECK: [[PFX]]float3x3_Val = [[STR]] [9 x float][[ZI]], align 4
//CHECK: [[PFX]]float3x4_Val = [[STR]] [12 x float][[ZI]], align 4
TYPE_DECL( float3x1 );
TYPE_DECL( float3x2 );
TYPE_DECL( float3x3 );
TYPE_DECL( float3x4 );

//CHECK: [[PFX]]float4x1_Val = [[STR]] [4 x float][[ZI]], align 4
//CHECK: [[PFX]]float4x2_Val = [[STR]] [8 x float][[ZI]], align 4
//CHECK: [[PFX]]float4x3_Val = [[STR]] [12 x float][[ZI]], align 4
//CHECK: [[PFX]]float4x4_Val = [[STR]] [16 x float][[ZI]], align 4
TYPE_DECL( float4x1 );
TYPE_DECL( float4x2 );
TYPE_DECL( float4x3 );
TYPE_DECL( float4x4 );

//CHECK: [[PFX]]double1x1_Val = [[STR]] [1 x double][[ZI]], align 8
//CHECK: [[PFX]]double1x2_Val = [[STR]] [2 x double][[ZI]], align 8
//CHECK: [[PFX]]double1x3_Val = [[STR]] [3 x double][[ZI]], align 8
//CHECK: [[PFX]]double1x4_Val = [[STR]] [4 x double][[ZI]], align 8
TYPE_DECL( double1x1 );
TYPE_DECL( double1x2 );
TYPE_DECL( double1x3 );
TYPE_DECL( double1x4 );

//CHECK: [[PFX]]double2x1_Val = [[STR]] [2 x double][[ZI]], align 8
//CHECK: [[PFX]]double2x2_Val = [[STR]] [4 x double][[ZI]], align 8
//CHECK: [[PFX]]double2x3_Val = [[STR]] [6 x double][[ZI]], align 8
//CHECK: [[PFX]]double2x4_Val = [[STR]] [8 x double][[ZI]], align 8
TYPE_DECL( double2x1 );
TYPE_DECL( double2x2 );
TYPE_DECL( double2x3 );
TYPE_DECL( double2x4 );

//CHECK: [[PFX]]double3x1_Val = [[STR]] [3 x double][[ZI]], align 8
//CHECK: [[PFX]]double3x2_Val = [[STR]] [6 x double][[ZI]], align 8
//CHECK: [[PFX]]double3x3_Val = [[STR]] [9 x double][[ZI]], align 8
//CHECK: [[PFX]]double3x4_Val = [[STR]] [12 x double][[ZI]], align 8
TYPE_DECL( double3x1 );
TYPE_DECL( double3x2 );
TYPE_DECL( double3x3 );
TYPE_DECL( double3x4 );

//CHECK: [[PFX]]double4x1_Val = [[STR]] [4 x double][[ZI]], align 8
//CHECK: [[PFX]]double4x2_Val = [[STR]] [8 x double][[ZI]], align 8
//CHECK: [[PFX]]double4x3_Val = [[STR]] [12 x double][[ZI]], align 8
//CHECK: [[PFX]]double4x4_Val = [[STR]] [16 x double][[ZI]], align 8
TYPE_DECL( double4x1 );
TYPE_DECL( double4x2 );
TYPE_DECL( double4x3 );
TYPE_DECL( double4x4 );

#ifndef SPIRV
}
#endif
