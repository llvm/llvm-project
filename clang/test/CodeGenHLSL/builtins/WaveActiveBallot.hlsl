// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call for int values.

// CHECK-LABEL: define {{.*}}test
uint4 test(bool p1) {
  // CHECK-SPIRV: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[SPIRVRET:.*]] = call spir_func <4 x i32> @llvm.spv.subgroup.ballot(i1 %{{[a-zA-Z0-9]+}}) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK-DXIL: %[[WAB:.*]] = call { i32, i32, i32, i32 } @llvm.dx.wave.ballot.i32(i1 %{{[a-zA-Z0-9]+}})
  // CHECK-DXIL-NEXT: extractvalue { i32, i32, i32, i32 } {{.*}} 0
  // CHECK-DXIL-NEXT: insertelement <4 x i32> poison, i32 {{.*}}, i32 0
  // CHECK-DXIL-NEXT: extractvalue { i32, i32, i32, i32 } {{.*}} 1
  // CHECK-DXIL-NEXT: insertelement <4 x i32> {{.*}}, i32 {{.*}}, i32 1
  // CHECK-DXIL-NEXT: extractvalue { i32, i32, i32, i32 } {{.*}} 2
  // CHECK-DXIL-NEXT: insertelement <4 x i32> {{.*}}, i32 {{.*}}, i32 2
  // CHECK-DXIL-NEXT: extractvalue { i32, i32, i32, i32 } {{.*}} 3
  // CHECK-DXIL-NEXT: %[[DXILRET:.*]] = insertelement <4 x i32> {{.*}}, i32 {{.*}}, i32 3
  // CHECK-DXIL-NEXT: ret <4 x i32> %[[DXILRET]]
  // CHECK-SPIRV: ret <4 x i32> %[[SPIRVRET]]

  return WaveActiveBallot(p1);
}
