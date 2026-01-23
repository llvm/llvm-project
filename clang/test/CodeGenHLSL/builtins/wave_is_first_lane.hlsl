// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL

[numthreads(1, 1, 1)]
void main() {
// CHECK-SPIRV: %[[#entry_tok:]] = call token @llvm.experimental.convergence.entry()

// CHECK-SPIRV: %[[#loop_tok:]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %[[#entry_tok]]) ]
  while (true) {

// CHECK-DXIL:  %[[#]] = call i1 @llvm.dx.wave.is.first.lane()
// CHECK-SPIRV: %[[#]] = call spir_func i1 @llvm.spv.wave.is.first.lane()
// CHECK-SPIRV-SAME: [ "convergencectrl"(token %[[#loop_tok]]) ]
    if (WaveIsFirstLane()) {
      break;
    }
  }

// CHECK-DXIL:  %[[#]] = call i1 @llvm.dx.wave.is.first.lane()
// CHECK-SPIRV: %[[#]] = call spir_func i1 @llvm.spv.wave.is.first.lane()
// CHECK-SPIRV-SAME: [ "convergencectrl"(token %[[#entry_tok]]) ]
  if (WaveIsFirstLane()) {
    return;
  }
}

// CHECK-DXIL:  i1 @llvm.dx.wave.is.first.lane() #[[#attr:]]
// CHECK-SPIRV: i1 @llvm.spv.wave.is.first.lane() #[[#attr:]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
