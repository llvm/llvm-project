// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -O3 -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK

// CHECK:  %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %splat.splat, <2 x float> %1)
// CHECK: ret float %dx.dot
float test_builtin_dot_float2_splat ( float p0, float2 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
}

// CHECK:  %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %splat.splat, <3 x float> %1)
// CHECK: ret float %dx.dot
float test_builtin_dot_float3_splat ( float p0, float3 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
}

// CHECK:  %dx.dot = call float @llvm.dx.dot.v4f32(<4 x float> %splat.splat, <4 x float> %1)
// CHECK: ret float %dx.dot
float test_builtin_dot_float4_splat ( float p0, float4 p1 ) {
  return __builtin_hlsl_dot( p0, p1 );
}

// CHECK: %conv = sitofp i32 %1 to float
// CHECK: %splat.splatinsert = insertelement <2 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %0, <2 x float> %splat.splat)
// CHECK: ret float %dx.dot
float test_dot_float2_int_splat ( float2 p0, int p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}

// CHECK: %conv = sitofp i32 %1 to float
// CHECK: %splat.splatinsert = insertelement <3 x float> poison, float %conv, i64 0
// CHECK: %splat.splat = shufflevector <3 x float> %splat.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK: %dx.dot = call float @llvm.dx.dot.v3f32(<3 x float> %0, <3 x float> %splat.splat)
// CHECK: ret float %dx.dot
float test_dot_float3_int_splat ( float3 p0, int p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}

// CHECK: %conv1 = uitofp i1 %tobool to double
// CHECK: %dx.dot = fmul double %conv, %conv1
// CHECK: %conv2 = fptrunc double %dx.dot to float
// CHECK: ret float %conv2
float builtin_bool_to_float_type_promotion ( float p0, bool p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}

// CHECK: %conv = uitofp i1 %tobool to double
// CHECK: %conv1 = fpext float %1 to double
// CHECK: %dx.dot = fmul double %conv, %conv1
// CHECK: %conv2 = fptrunc double %dx.dot to float
// CHECK: ret float %conv2
float builtin_bool_to_float_arg1_type_promotion ( bool p0, float p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}

// CHECK: %conv = zext i1 %tobool to i32
// CHECK: %conv3 = zext i1 %tobool2 to i32
// CHECK: %dx.dot = mul i32 %conv, %conv3
// CHECK: ret i32 %dx.dot
int test_builtin_dot_bool_type_promotion ( bool p0, bool p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}

// CHECK: %conv = fpext float %0 to double
// CHECK: %conv1 = sitofp i32 %1 to double
// CHECK: dx.dot = fmul double %conv, %conv1
// CHECK: %conv2 = fptrunc double %dx.dot to float
// CHECK: ret float %conv2
float test_builtin_dot_int_to_float_promotion ( float p0, int p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}


// CHECK: %conv = sitofp <2 x i32> %0 to <2 x float>
// CHECK: %splat.splat = shufflevector <2 x float> %splat.splatinsert, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK: %dx.dot = call float @llvm.dx.dot.v2f32(<2 x float> %conv, <2 x float> %splat.splat)
// CHECK: ret float %dx.dot
float test_builtin_dot_int_vect_to_float_vec_promotion ( int2 p0, float p1 ) {
  return __builtin_hlsl_dot ( p0, p1 );
}
