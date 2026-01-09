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
  // CHECK-SPIRV: %[[RET:.*]] = call spir_func <4 x i32> @llvm.spv.wave.ballot(i1 %{{[a-zA-Z0-9]+}}) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK-DXIL: %[[RETVAL:.*]] = alloca <4 x i32>, align 16
  // CHECK-DXIL: %[[WAB:.*]] = call { i32, i32, i32, i32 } @llvm.dx.wave.ballot.i32(i1 %{{[a-zA-Z0-9]+}})
  // CHECK-DXIL: store { i32, i32, i32, i32 } %[[WAB]], ptr %[[RETVAL]], align 16
  // CHECK-DXIL: %[[LOAD:.*]] = load <4 x i32>, ptr %[[RETVAL]], align 16
  // CHECK-DXIL: ret <4 x i32> %[[LOAD]]
  // CHECK-SPIRV: ret <4 x i32> %[[RET]]

  return WaveActiveBallot(p1);
}
