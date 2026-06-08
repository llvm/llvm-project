// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type -emit-llvm -O1 -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-vulkan-library %s -fnative-half-type -fnative-int16-type -emit-llvm -O0 -o - | FileCheck %s --check-prefix=SPIRV



// CHECK: define {{.*}} <4 x i32> {{.*}}test_mat2x2{{.*}}(<4 x double> {{.*}} [[VALD:%.*]])
// CHECK:      [[VALRET:%.*]] = {{.*}} call { <4 x i32>, <4 x i32> } @llvm.dx.splitdouble.v4i32(<4 x double> [[VALD]])
// CHECK-NEXT: extractvalue { <4 x i32>, <4 x i32> } [[VALRET]], 0
// CHECK-NEXT: extractvalue { <4 x i32>, <4 x i32> } [[VALRET]], 1
//
// SPIRV: define hidden spir_func {{.*}} <4 x i32> {{.*}}test_mat2x2{{.*}}(<4 x double> {{.*}} [[VALD:%.*]])
// SPIRV-NOT:  @llvm.dx.splitdouble
// SPIRV:      [[LOAD:%.*]] = load <4 x double>, ptr [[VALD]].addr, align {{[0-9]+}}
// SPIRV-NEXT: [[CAST:%.*]] = bitcast <4 x double> [[LOAD]] to <8 x i32>
// SPIRV-NEXT: [[SHUF1:%.*]] = shufflevector <8 x i32> [[CAST]], <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
// SPIRV-NEXT: [[SHUF2:%.*]] = shufflevector <8 x i32> [[CAST]], <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
uint2x2 test_mat2x2(double2x2 D) {
  uint2x2 A, B;
  asuint(D, A, B);
  return A + B;
}

// CHECK: define {{.*}} <6 x i32> {{.*}}test_mat2x3{{.*}}(<6 x double> {{.*}} [[VALD:%.*]])
// CHECK:      [[VALRET:%.*]] = {{.*}} call { <6 x i32>, <6 x i32> } @llvm.dx.splitdouble.v6i32(<6 x double> [[VALD]])
// CHECK-NEXT: extractvalue { <6 x i32>, <6 x i32> } [[VALRET]], 0
// CHECK-NEXT: extractvalue { <6 x i32>, <6 x i32> } [[VALRET]], 1
//
// SPIRV: define hidden spir_func {{.*}} <6 x i32> {{.*}}test_mat2x3{{.*}}(<6 x double> {{.*}} [[VALD:%.*]])
// SPIRV-NOT:  @llvm.dx.splitdouble
// SPIRV:      [[LOAD:%.*]] = load <6 x double>, ptr [[VALD]].addr, align {{[0-9]+}}
// SPIRV-NEXT: [[CAST:%.*]] = bitcast <6 x double> [[LOAD]] to <12 x i32>
// SPIRV-NEXT: [[SHUF1:%.*]] = shufflevector <12 x i32> [[CAST]], <12 x i32> poison, <6 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10>
// SPIRV-NEXT: [[SHUF2:%.*]] = shufflevector <12 x i32> [[CAST]], <12 x i32> poison, <6 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11>
uint2x3 test_mat2x3(double2x3 D) {
  uint2x3 A, B;
  asuint(D, A, B);
  return A + B;
}

// CHECK: define {{.*}} <9 x i32> {{.*}}test_mat3x3{{.*}}(<9 x double> {{.*}} [[VALD:%.*]])
// CHECK:      [[VALRET:%.*]] = {{.*}} call { <9 x i32>, <9 x i32> } @llvm.dx.splitdouble.v9i32(<9 x double> [[VALD]])
// CHECK-NEXT: extractvalue { <9 x i32>, <9 x i32> } [[VALRET]], 0
// CHECK-NEXT: extractvalue { <9 x i32>, <9 x i32> } [[VALRET]], 1
//
// SPIRV: define hidden spir_func {{.*}} <9 x i32> {{.*}}test_mat3x3{{.*}}(<9 x double> {{.*}} [[VALD:%.*]])
// SPIRV-NOT:  @llvm.dx.splitdouble
// SPIRV:      [[LOAD:%.*]] = load <9 x double>, ptr [[VALD]].addr, align {{[0-9]+}}
// SPIRV-NEXT: [[CAST:%.*]] = bitcast <9 x double> [[LOAD]] to <18 x i32>
// SPIRV-NEXT: [[SHUF1:%.*]] = shufflevector <18 x i32> [[CAST]], <18 x i32> poison, <9 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16>
// SPIRV-NEXT: [[SHUF2:%.*]] = shufflevector <18 x i32> [[CAST]], <18 x i32> poison, <9 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17>
uint3x3 test_mat3x3(double3x3 D) {
  uint3x3 A, B;
  asuint(D, A, B);
  return A + B;
}

// CHECK: define {{.*}} <16 x i32> {{.*}}test_mat4x4{{.*}}(<16 x double> {{.*}} [[VALD:%.*]])
// CHECK:      [[VALRET:%.*]] = {{.*}} call { <16 x i32>, <16 x i32> } @llvm.dx.splitdouble.v16i32(<16 x double> [[VALD]])
// CHECK-NEXT: extractvalue { <16 x i32>, <16 x i32> } [[VALRET]], 0
// CHECK-NEXT: extractvalue { <16 x i32>, <16 x i32> } [[VALRET]], 1
//
// SPIRV: define hidden spir_func {{.*}} <16 x i32> {{.*}}test_mat4x4{{.*}}(<16 x double> {{.*}} [[VALD:%.*]])
// SPIRV-NOT:  @llvm.dx.splitdouble
// SPIRV:      [[LOAD:%.*]] = load <16 x double>, ptr [[VALD]].addr, align {{[0-9]+}}
// SPIRV-NEXT: [[CAST:%.*]] = bitcast <16 x double> [[LOAD]] to <32 x i32>
// SPIRV-NEXT: [[SHUF1:%.*]] = shufflevector <32 x i32> [[CAST]], <32 x i32> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
// SPIRV-NEXT: [[SHUF2:%.*]] = shufflevector <32 x i32> [[CAST]], <32 x i32> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
uint4x4 test_mat4x4(double4x4 D) {
  uint4x4 A, B;
  asuint(D, A, B);
  return A + B;
}
