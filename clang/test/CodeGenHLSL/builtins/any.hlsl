// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_int16_t(int16_t p0) { return any(p0); }
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v2i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_int16_t2(int16_t2 p0) { return any(p0); }
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v3i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_int16_t3(int16_t3 p0) { return any(p0); }
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v4i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_int16_t4(int16_t4 p0) { return any(p0); }

// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_uint16_t(uint16_t p0) { return any(p0); }
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v2i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_uint16_t2(uint16_t2 p0) { return any(p0); }
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v3i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_uint16_t3(uint16_t3 p0) { return any(p0); }
// NATIVE_HALF: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v4i16
// NATIVE_HALF: ret i1 %dx.any
bool test_any_uint16_t4(uint16_t4 p0) { return any(p0); }
#endif // __HLSL_ENABLE_16_BIT

// CHECK: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.f16
// NO_HALF: %dx.any = call i1 @llvm.dx.any.f32
// CHECK: ret i1 %dx.any
bool test_any_half(half p0) { return any(p0); }

// CHECK: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v2f16
// NO_HALF: %dx.any = call i1 @llvm.dx.any.v2f32
// CHECK: ret i1 %dx.any
bool test_any_half2(half2 p0) { return any(p0); }

// CHECK: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v3f16
// NO_HALF: %dx.any = call i1 @llvm.dx.any.v3f32
// CHECK: ret i1 %dx.any
bool test_any_half3(half3 p0) { return any(p0); }

// CHECK: define noundef i1 @
// NATIVE_HALF: %dx.any = call i1 @llvm.dx.any.v4f16
// NO_HALF: %dx.any = call i1 @llvm.dx.any.v4f32
// CHECK: ret i1 %dx.any
bool test_any_half4(half4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.f32
// CHECK: ret i1 %dx.any
bool test_any_float(float p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2f32
// CHECK: ret i1 %dx.any
bool test_any_float2(float2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3f32
// CHECK: ret i1 %dx.any
bool test_any_float3(float3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4f32
// CHECK: ret i1 %dx.any
bool test_any_float4(float4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.f64
// CHECK: ret i1 %dx.any
bool test_any_double(double p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2f64
// CHECK: ret i1 %dx.any
bool test_any_double2(double2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3f64
// CHECK: ret i1 %dx.any
bool test_any_double3(double3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4f64
// CHECK: ret i1 %dx.any
bool test_any_double4(double4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.i32
// CHECK: ret i1 %dx.any
bool test_any_int(int p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2i32
// CHECK: ret i1 %dx.any
bool test_any_int2(int2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3i32
// CHECK: ret i1 %dx.any
bool test_any_int3(int3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4i32
// CHECK: ret i1 %dx.any
bool test_any_int4(int4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.i32
// CHECK: ret i1 %dx.any
bool test_any_uint(uint p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2i32
// CHECK: ret i1 %dx.any
bool test_any_uint2(uint2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3i32
// CHECK: ret i1 %dx.any
bool test_any_uint3(uint3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4i32
// CHECK: ret i1 %dx.any
bool test_any_uint4(uint4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.i64
// CHECK: ret i1 %dx.any
bool test_any_int64_t(int64_t p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2i64
// CHECK: ret i1 %dx.any
bool test_any_int64_t2(int64_t2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3i64
// CHECK: ret i1 %dx.any
bool test_any_int64_t3(int64_t3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4i64
// CHECK: ret i1 %dx.any
bool test_any_int64_t4(int64_t4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.i64
// CHECK: ret i1 %dx.any
bool test_any_uint64_t(uint64_t p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2i64
// CHECK: ret i1 %dx.any
bool test_any_uint64_t2(uint64_t2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3i64
// CHECK: ret i1 %dx.any
bool test_any_uint64_t3(uint64_t3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4i64
// CHECK: ret i1 %dx.any
bool test_any_uint64_t4(uint64_t4 p0) { return any(p0); }

// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.i1
// CHECK: ret i1 %dx.any
bool test_any_bool(bool p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v2i1
// CHECK: ret i1 %dx.any
bool test_any_bool2(bool2 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v3i1
// CHECK: ret i1 %dx.any
bool test_any_bool3(bool3 p0) { return any(p0); }
// CHECK: define noundef i1 @
// CHECK: %dx.any = call i1 @llvm.dx.any.v4i1
// CHECK: ret i1 %dx.any
bool test_any_bool4(bool4 p0) { return any(p0); }
