// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL

[numthreads(1, 1, 1)]
void main() {
  uint a, b;
// CHECK-SPIRV: %[[#entry_tok:]] = call token @llvm.experimental.convergence.entry()

// CHECK-SPIRV: %[[#loop_tok:]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %[[#entry_tok]]) ]
  while (a) {

// CHECK-DXIL:  %[[#]] = call i32 @llvm.dx.wave.get.lane.count()
// CHECK-SPIRV: %[[#]] = call spir_func i32 @llvm.spv.wave.get.lane.count()
// CHECK-SPIRV-SAME: [ "convergencectrl"(token %[[#loop_tok]]) ]
    a = WaveGetLaneCount();
  }

// CHECK-DXIL:  %[[#]] = call i32 @llvm.dx.wave.get.lane.count()
// CHECK-SPIRV: %[[#]] = call spir_func i32 @llvm.spv.wave.get.lane.count()
// CHECK-SPIRV-SAME: [ "convergencectrl"(token %[[#entry_tok]]) ]
  b = WaveGetLaneCount();
}

// CHECK-DXIL:  i32 @llvm.dx.wave.get.lane.count() #[[#attr:]]
// CHECK-SPIRV: i32 @llvm.spv.wave.get.lane.count() #[[#attr:]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}

