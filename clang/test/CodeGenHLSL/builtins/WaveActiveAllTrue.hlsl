// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call for int values.

// CHECK-LABEL: define {{.*}}test
bool test(bool p1) {
  // CHECK-SPIRV: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func i1 @llvm.spv.wave.all(i1 %{{[a-zA-Z0-9]+}}) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call i1 @llvm.dx.wave.all(i1 %{{[a-zA-Z0-9]+}})
  // CHECK:  ret i1 %[[RET]]
  return WaveActiveAllTrue(p1);
}
