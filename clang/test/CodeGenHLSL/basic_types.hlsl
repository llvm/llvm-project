// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - -DNAMESPACED| FileCheck %s


// CHECK:"?uint16_t_Val@@3GA" = global i16 0, align 2
// CHECK:"?int16_t_Val@@3FA" = global i16 0, align 2
// CHECK:"?uint_Val@@3IA" = global i32 0, align 4
// CHECK:"?uint64_t_Val@@3KA" = global i64 0, align 8
// CHECK:"?int64_t_Val@@3JA" = global i64 0, align 8
// CHECK:"?int16_t2_Val@@3T?$__vector@F$01@__clang@@A" = global <2 x i16> zeroinitializer, align 4
// CHECK:"?int16_t3_Val@@3T?$__vector@F$02@__clang@@A" = global <3 x i16> zeroinitializer, align 8
// CHECK:"?int16_t4_Val@@3T?$__vector@F$03@__clang@@A" = global <4 x i16> zeroinitializer, align 8
// CHECK:"?uint16_t2_Val@@3T?$__vector@G$01@__clang@@A" = global <2 x i16> zeroinitializer, align 4
// CHECK:"?uint16_t3_Val@@3T?$__vector@G$02@__clang@@A" = global <3 x i16> zeroinitializer, align 8
// CHECK:"?uint16_t4_Val@@3T?$__vector@G$03@__clang@@A" = global <4 x i16> zeroinitializer, align 8
// CHECK:"?int2_Val@@3T?$__vector@H$01@__clang@@A" = global <2 x i32> zeroinitializer, align 8
// CHECK:"?int3_Val@@3T?$__vector@H$02@__clang@@A" = global <3 x i32> zeroinitializer, align 16
// CHECK:"?int4_Val@@3T?$__vector@H$03@__clang@@A" = global <4 x i32> zeroinitializer, align 16
// CHECK:"?uint2_Val@@3T?$__vector@I$01@__clang@@A" = global <2 x i32> zeroinitializer, align 8
// CHECK:"?uint3_Val@@3T?$__vector@I$02@__clang@@A" = global <3 x i32> zeroinitializer, align 16
// CHECK:"?uint4_Val@@3T?$__vector@I$03@__clang@@A" = global <4 x i32> zeroinitializer, align 16
// CHECK:"?int64_t2_Val@@3T?$__vector@J$01@__clang@@A" = global <2 x i64> zeroinitializer, align 16
// CHECK:"?int64_t3_Val@@3T?$__vector@J$02@__clang@@A" = global <3 x i64> zeroinitializer, align 32
// CHECK:"?int64_t4_Val@@3T?$__vector@J$03@__clang@@A" = global <4 x i64> zeroinitializer, align 32
// CHECK:"?uint64_t2_Val@@3T?$__vector@K$01@__clang@@A" = global <2 x i64> zeroinitializer, align 16
// CHECK:"?uint64_t3_Val@@3T?$__vector@K$02@__clang@@A" = global <3 x i64> zeroinitializer, align 32
// CHECK:"?uint64_t4_Val@@3T?$__vector@K$03@__clang@@A" = global <4 x i64> zeroinitializer, align 32
// CHECK:"?half2_Val@@3T?$__vector@$f16@$01@__clang@@A" = global <2 x half> zeroinitializer, align 4
// CHECK:"?half3_Val@@3T?$__vector@$f16@$02@__clang@@A" = global <3 x half> zeroinitializer, align 8
// CHECK:"?half4_Val@@3T?$__vector@$f16@$03@__clang@@A" = global <4 x half> zeroinitializer, align 8
// CHECK:"?float2_Val@@3T?$__vector@M$01@__clang@@A" = global <2 x float> zeroinitializer, align 8
// CHECK:"?float3_Val@@3T?$__vector@M$02@__clang@@A" = global <3 x float> zeroinitializer, align 16
// CHECK:"?float4_Val@@3T?$__vector@M$03@__clang@@A" = global <4 x float> zeroinitializer, align 16
// CHECK:"?double2_Val@@3T?$__vector@N$01@__clang@@A" = global <2 x double> zeroinitializer, align 16
// CHECK:"?double3_Val@@3T?$__vector@N$02@__clang@@A" = global <3 x double> zeroinitializer, align 32
// CHECK:"?double4_Val@@3T?$__vector@N$03@__clang@@A" = global <4 x double> zeroinitializer, align 32

#ifdef NAMESPACED
#define TYPE_DECL(T)  hlsl::T T##_Val
#else
#define TYPE_DECL(T)  T T##_Val
#endif

#ifdef __HLSL_ENABLE_16_BIT
TYPE_DECL(uint16_t);
TYPE_DECL(int16_t);
#endif

// unsigned 32-bit integer.
TYPE_DECL(uint);

// 64-bit integer.
TYPE_DECL(uint64_t);
TYPE_DECL(int64_t);

// built-in vector data types:

#ifdef __HLSL_ENABLE_16_BIT
TYPE_DECL(int16_t2   );
TYPE_DECL(int16_t3   );
TYPE_DECL(int16_t4   );
TYPE_DECL( uint16_t2 );
TYPE_DECL( uint16_t3 );
TYPE_DECL( uint16_t4 );
#endif

TYPE_DECL( int2  );
TYPE_DECL( int3  );
TYPE_DECL( int4  );
TYPE_DECL( uint2 );
TYPE_DECL( uint3 );
TYPE_DECL( uint4     );
TYPE_DECL( int64_t2  );
TYPE_DECL( int64_t3  );
TYPE_DECL( int64_t4  );
TYPE_DECL( uint64_t2 );
TYPE_DECL( uint64_t3 );
TYPE_DECL( uint64_t4 );

#ifdef __HLSL_ENABLE_16_BIT
TYPE_DECL(half2 );
TYPE_DECL(half3 );
TYPE_DECL(half4 );
#endif

TYPE_DECL( float2  );
TYPE_DECL( float3  );
TYPE_DECL( float4  );
TYPE_DECL( double2 );
TYPE_DECL( double3 );
TYPE_DECL( double4 );
