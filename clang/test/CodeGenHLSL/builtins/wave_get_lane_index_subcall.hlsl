// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK: define spir_func noundef i32 @_Z6test_1v() [[A0:#[0-9]+]] {
// CHECK: %[[C1:[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: call i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %[[C1]]) ]
uint test_1() {
  return WaveGetLaneIndex();
}

// CHECK-DAG: declare i32 @__hlsl_wave_get_lane_index() [[A1:#[0-9]+]]

// CHECK: define spir_func noundef i32 @_Z6test_2v() [[A0]] {
// CHECK: %[[C2:[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: call spir_func noundef i32 @_Z6test_1v() [ "convergencectrl"(token %[[C2]]) ]
uint test_2() {
  return test_1();
}

// CHECK-DAG: attributes [[A0]] = {{{.*}}convergent{{.*}}}
// CHECK-DAG: attributes [[A1]] = {{{.*}}convergent{{.*}}}
