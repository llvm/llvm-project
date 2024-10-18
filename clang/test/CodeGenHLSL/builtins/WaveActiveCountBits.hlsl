// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_bool
int test_bool(bool expr) {
  // CHECK-SPIRV: %[[#entry_tok:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func i32 @llvm.spv.wave.active.countbits(i1 %{{.*}}) [ "convergencectrl"(token %[[#entry_tok]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call i32 @llvm.dx.wave.active.countbits(i1 %{{.*}})
  // CHECK:  ret i32 %[[RET]]
  return WaveActiveCountBits(expr);
}

// CHECK-DXIL: declare i32 @llvm.dx.wave.active.countbits(i1) #[[#attr:]]
// CHECK-SPIRV: declare i32 @llvm.spv.wave.active.countbits(i1) #[[#attr:]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
