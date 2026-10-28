// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefix=CHECK-DXIL

[numthreads(1, 1, 1)]
void main() {
  uint a, b;

  while (a) {

// CHECK-DXIL:  %[[#]] = call i32 @llvm.dx.wave.get.lane.count()
// CHECK-SPIRV: %[[#]] = call i32 @llvm.spv.subgroup.size()
    a = WaveGetLaneCount();
  }

// CHECK-DXIL:  %[[#]] = call i32 @llvm.dx.wave.get.lane.count()
// CHECK-SPIRV: %[[#]] = call i32 @llvm.spv.subgroup.size()
  b = WaveGetLaneCount();
}

// CHECK-DXIL:  i32 @llvm.dx.wave.get.lane.count() #[[#attr:]]

// CHECK-DXIL: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
