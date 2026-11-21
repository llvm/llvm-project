// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXCHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,SPVCHECK

// DXCHECK: define hidden [[FN_TYPE:]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f1x2{{.*}}(
// SPVCHECK: define hidden [[FN_TYPE:spir_func ]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f1x2{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <2 x float>, <2 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <2 x float>, <2 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<2 x i32> [[EXPI]] to <2 x float>
// CHECK-NEXT: store <2 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <2 x float> [[MANT]]
float1x2 test_frexp_f1x2(float1x2 X, out float1x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <3 x float> @{{.*}}test_frexp_f1x3{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <3 x float>, <3 x i32> } @llvm.frexp.v3f32.v3i32(<3 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <3 x float>, <3 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <3 x float>, <3 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<3 x i32> [[EXPI]] to <3 x float>
// CHECK-NEXT: store <3 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <3 x float> [[MANT]]
float1x3 test_frexp_f1x3(float1x3 X, out float1x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f1x4{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<4 x i32> [[EXPI]] to <4 x float>
// CHECK-NEXT: store <4 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <4 x float> [[MANT]]
float1x4 test_frexp_f1x4(float1x4 X, out float1x4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <2 x float> @{{.*}}test_frexp_f2x1{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <2 x float>, <2 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <2 x float>, <2 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<2 x i32> [[EXPI]] to <2 x float>
// CHECK-NEXT: store <2 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <2 x float> [[MANT]]
float2x1 test_frexp_f2x1(float2x1 X, out float2x1 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f2x2{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<4 x i32> [[EXPI]] to <4 x float>
// CHECK-NEXT: store <4 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <4 x float> [[MANT]]
float2x2 test_frexp_f2x2(float2x2 X, out float2x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <6 x float> @{{.*}}test_frexp_f2x3{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <6 x float>, <6 x i32> } @llvm.frexp.v6f32.v6i32(<6 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <6 x float>, <6 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <6 x float>, <6 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<6 x i32> [[EXPI]] to <6 x float>
// CHECK-NEXT: store <6 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <6 x float> [[MANT]]
float2x3 test_frexp_f2x3(float2x3 X, out float2x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <8 x float> @{{.*}}test_frexp_f2x4{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <8 x float>, <8 x i32> } @llvm.frexp.v8f32.v8i32(<8 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <8 x float>, <8 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <8 x float>, <8 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<8 x i32> [[EXPI]] to <8 x float>
// CHECK-NEXT: store <8 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <8 x float> [[MANT]]
float2x4 test_frexp_f2x4(float2x4 X, out float2x4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <3 x float> @{{.*}}test_frexp_f3x1{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <3 x float>, <3 x i32> } @llvm.frexp.v3f32.v3i32(<3 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <3 x float>, <3 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <3 x float>, <3 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<3 x i32> [[EXPI]] to <3 x float>
// CHECK-NEXT: store <3 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <3 x float> [[MANT]]
float3x1 test_frexp_f3x1(float3x1 X, out float3x1 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <6 x float> @{{.*}}test_frexp_f3x2{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <6 x float>, <6 x i32> } @llvm.frexp.v6f32.v6i32(<6 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <6 x float>, <6 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <6 x float>, <6 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<6 x i32> [[EXPI]] to <6 x float>
// CHECK-NEXT: store <6 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <6 x float> [[MANT]]
float3x2 test_frexp_f3x2(float3x2 X, out float3x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <9 x float> @{{.*}}test_frexp_f3x3{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <9 x float>, <9 x i32> } @llvm.frexp.v9f32.v9i32(<9 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <9 x float>, <9 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <9 x float>, <9 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<9 x i32> [[EXPI]] to <9 x float>
// CHECK-NEXT: store <9 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <9 x float> [[MANT]]
float3x3 test_frexp_f3x3(float3x3 X, out float3x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <12 x float> @{{.*}}test_frexp_f3x4{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <12 x float>, <12 x i32> } @llvm.frexp.v12f32.v12i32(<12 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <12 x float>, <12 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <12 x float>, <12 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<12 x i32> [[EXPI]] to <12 x float>
// CHECK-NEXT: store <12 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <12 x float> [[MANT]]
float3x4 test_frexp_f3x4(float3x4 X, out float3x4 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <4 x float> @{{.*}}test_frexp_f4x1{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <4 x float>, <4 x i32> } @llvm.frexp.v4f32.v4i32(<4 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <4 x float>, <4 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<4 x i32> [[EXPI]] to <4 x float>
// CHECK-NEXT: store <4 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <4 x float> [[MANT]]
float4x1 test_frexp_f4x1(float4x1 X, out float4x1 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <8 x float> @{{.*}}test_frexp_f4x2{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <8 x float>, <8 x i32> } @llvm.frexp.v8f32.v8i32(<8 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <8 x float>, <8 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <8 x float>, <8 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<8 x i32> [[EXPI]] to <8 x float>
// CHECK-NEXT: store <8 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <8 x float> [[MANT]]
float4x2 test_frexp_f4x2(float4x2 X, out float4x2 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <12 x float> @{{.*}}test_frexp_f4x3{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <12 x float>, <12 x i32> } @llvm.frexp.v12f32.v12i32(<12 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <12 x float>, <12 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <12 x float>, <12 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<12 x i32> [[EXPI]] to <12 x float>
// CHECK-NEXT: store <12 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <12 x float> [[MANT]]
float4x3 test_frexp_f4x3(float4x3 X, out float4x3 E) { return frexp(X, E); }

// CHECK: define hidden [[FN_TYPE]]noundef nofpclass(nan inf) <16 x float> @{{.*}}test_frexp_f4x4{{.*}}(
// CHECK:      [[FREXP:%.*]] = call { <16 x float>, <16 x i32> } @llvm.frexp.v16f32.v16i32(<16 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <16 x float>, <16 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <16 x float>, <16 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<16 x i32> [[EXPI]] to <16 x float>
// CHECK-NEXT: store <16 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK:      ret <16 x float> [[MANT]]
float4x4 test_frexp_f4x4(float4x4 X, out float4x4 E) { return frexp(X, E); }

// int exponent
// CHECK-LABEL: define {{.*}}test_frexp_int_exp_matrix
// CHECK:      [[FREXP:%.*]] = call { <6 x float>, <6 x i32> } @llvm.frexp.v6f32.v6i32(<6 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <6 x float>, <6 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <6 x float>, <6 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<6 x i32> [[EXPI]] to <6 x float>
// CHECK-NEXT: store <6 x float> [[EXPF]], ptr [[TMP:%.*]], align 4
// CHECK-NEXT: [[LOAD:%.*]] = load <6 x float>, ptr [[TMP]], align 4
// CHECK-NEXT: [[CONV:%.*]] = fptosi <6 x float> [[LOAD]] to <6 x i32>
// CHECK-NEXT: store <6 x i32> [[CONV]], ptr %{{.*}}, align 4
float2x3 test_frexp_int_exp_matrix(float2x3 X, out int2x3 E) {
  return frexp(X, E);
}

// Direct calls to the builtin store the exponent straight to the lvalue.
// CHECK-LABEL: test_frexp_builtin_f3x3
// CHECK:      [[FREXP:%.*]] = call { <9 x float>, <9 x i32> } @llvm.frexp.v9f32.v9i32(<9 x float> %{{.*}})
// CHECK-NEXT: [[MANT:%.*]] = extractvalue { <9 x float>, <9 x i32> } [[FREXP]], 0
// CHECK-NEXT: [[EXPI:%.*]] = extractvalue { <9 x float>, <9 x i32> } [[FREXP]], 1
// CHECK-NEXT: [[EXPF:%.*]] = sitofp {{.*}}<9 x i32> [[EXPI]] to <9 x float>
// CHECK-NEXT: store <9 x float> [[EXPF]], ptr [[E:%.*]], align 4
// CHECK:      ret <9 x float> [[MANT]]
float3x3 test_frexp_builtin_f3x3(float3x3 X) {
  float3x3 E;
  return __builtin_hlsl_elementwise_frexp(X, E);
}

