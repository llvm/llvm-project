// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=spv

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_bool
int test_bool(bool expr) {
  // CHECK:  call {{.*}} @llvm.[[TARGET]].wave.active.countbits
  return WaveActiveCountBits(expr);
}

// CHECK: declare i32 @llvm.[[TARGET]].wave.active.countbits(i1) #[[#attr:]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
