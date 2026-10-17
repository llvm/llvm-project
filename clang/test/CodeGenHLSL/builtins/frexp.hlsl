// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK

// DXCHECK: define hidden noundef nofpclass(nan inf) float @_Z12test_frexp_ffRf(
// SPVCHECK: define hidden spir_func noundef nofpclass(nan inf) float @_Z12test_frexp_ffRf(
// CHECK:      [[X:%.*]] = load float, ptr %{{.*}}
// CHECK:      [[FREXP:%.*]] = call { float, i32 } @llvm.frexp.f32.i32(float [[X]])
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { float, i32 } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { float, i32 } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}i32 [[EXPI]] to float
// CHECK-NEXT: store float [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK-NEXT: [[WB:%.*]] = load float, ptr [[TMP]], align 4
// CHECK-NEXT: store float [[WB]], ptr %{{.*}}, align 4
// CHECK:      ret float [[MANT]]
float test_frexp_f(float X, out float E) { return frexp(X, E); }

// DXCHECK: define hidden noundef nofpclass(nan inf) <2 x float> @_Z13test_frexp_f2Dv2_fRS_(
// SPVCHECK: define hidden spir_func noundef nofpclass(nan inf) <2 x float> @_Z13test_frexp_f2Dv2_fRS_(
// CHECK:      [[FREXP:%.*]] = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <2 x float>, <2 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <2 x float>, <2 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<2 x i32> [[EXPI]] to <2 x float>
// CHECK-NEXT: store <2 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <2 x float> [[MANT]]
float2 test_frexp_f2(float2 X, out float2 E) { return frexp(X, E); }

// DXCHECK: define hidden noundef nofpclass(nan inf) <3 x float> @_Z13test_frexp_f3Dv3_fRS_(
// SPVCHECK: define hidden spir_func noundef nofpclass(nan inf) <3 x float> @_Z13test_frexp_f3Dv3_fRS_(
// CHECK:      [[FREXP:%.*]] = call { <3 x float>, <3 x i32> } @llvm.frexp.v3f32.v3i32(<3 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <3 x float>, <3 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <3 x float>, <3 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<3 x i32> [[EXPI]] to <3 x float>
// CHECK-NEXT: store <3 x float> [[EXPF]], ptr [[TMP:%.*]]
// CHECK:      ret <3 x float> [[MANT]]
float3 test_frexp_f3(float3 X, out float3 E) { return frexp(X, E); }

// DXCHECK: define hidden noundef nofpclass(nan inf) <4 x float> @_Z13test_frexp_f4Dv4_fRS_(
// SPVCHECK: define hidden spir_func noundef nofpclass(nan inf) <4 x float> @_Z13test_frexp_f4Dv4_fRS_(
// CHECK:      [[FREXP:%.*]] = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<4 x i32> [[EXPI]] to <4 x float>
// CHECK-NEXT: store <4 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <4 x float> [[MANT]]
float4 test_frexp_f4(float4 X, out float4 E) { return frexp(X, E); }

// DXCHECK: define hidden noundef nofpclass(nan inf) <5 x float> @_Z13test_frexp_f5Dv5_fRS_(
// SPVCHECK: define hidden spir_func noundef nofpclass(nan inf) <5 x float> @_Z13test_frexp_f5Dv5_fRS_(
// CHECK:      [[FREXP:%.*]] = call { <5 x float>, <5 x i32> } @llvm.frexp.v5f32.v5i32(<5 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <5 x float>, <5 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <5 x float>, <5 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<5 x i32> [[EXPI]] to <5 x float>
// CHECK-NEXT: store <5 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <5 x float> [[MANT]]
vector<float, 5> test_frexp_f5(vector<float, 5> X, out vector<float, 5> E) {
  return frexp(X, E);
}

// Direct calls to the builtin store the exponent straight to the lvalue.
// CHECK-LABEL: test_frexp_builtin
// CHECK:      [[FREXP:%.*]] = call { float, i32 } @llvm.frexp.f32.i32(float %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { float, i32 } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { float, i32 } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}i32 [[EXPI]] to float
// CHECK-NEXT: store float [[EXPF]], ptr [[E:%.*]], align 4
// CHECK-NEXT: store float [[MANT]], ptr [[M:%.*]], align 4
// CHECK-NEXT: [[MLOAD:%.*]] = load float, ptr [[M]], align 4
// CHECK-NEXT: [[ELOAD:%.*]] = load float, ptr [[E]], align 4
// CHECK-NEXT: fadd {{.*}}float [[MLOAD]], [[ELOAD]]
float test_frexp_builtin(float X) {
  float E;
  float M = __builtin_hlsl_elementwise_frexp(X, E);
  return M + E;
}

// The exponent can also be an integer, as with DXC. It is computed as a float
// and converted when it is written back.

// CHECK-LABEL: define {{.*}}test_frexp_int_exp_scalar
// CHECK:      [[FREXP:%.*]] = call { float, i32 } @llvm.frexp.f32.i32(float %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { float, i32 } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { float, i32 } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}i32 [[EXPI]] to float
// CHECK-NEXT: store float [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK-NEXT: [[LOAD:%.*]] = load float, ptr [[TMP]], align 4
// CHECK-NEXT: [[CONV:%.*]] = fptosi float [[LOAD]] to i32
// CHECK-NEXT: store i32 [[CONV]], ptr %{{.*}}, align 4
// CHECK:      ret float [[MANT]]
float test_frexp_int_exp_scalar(float X, out int E) { return frexp(X, E); }

// CHECK-LABEL: define {{.*}}test_frexp_int_exp_long
// CHECK:      [[FREXP:%.*]] = call { <8 x float>, <8 x i32> } @llvm.frexp.v8f32.v8i32(<8 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <8 x float>, <8 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <8 x float>, <8 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<8 x i32> [[EXPI]] to <8 x float>
// CHECK-NEXT: store <8 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK-NEXT: [[LOAD:%.*]] = load <8 x float>, ptr [[TMP]], align 4
// CHECK-NEXT: [[CONV:%.*]] = fptosi <8 x float> [[LOAD]] to <8 x i32>
// CHECK-NEXT: store <8 x i32> [[CONV]], ptr %{{.*}}, align 4
// CHECK:      ret <8 x float> [[MANT]]
vector<float, 8> test_frexp_int_exp_long(vector<float, 8> X,
                                         out vector<int, 8> E) {
  return frexp(X, E);
}

