// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define spir_func noundef i32 @_Z6test_1v() [[A0:#[0-9]+]] {
// CHECK: %[[CI:[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %[[CI]]) ]
uint test_1() {
  return WaveGetLaneIndex();
}

// CHECK: declare i32 @__hlsl_wave_get_lane_index() [[A1:#[0-9]+]]

// CHECK-DAG: attributes [[A0]] = { {{.*}}convergent{{.*}} }
// CHECK-DAG: attributes [[A1]] = { {{.*}}convergent{{.*}} }
