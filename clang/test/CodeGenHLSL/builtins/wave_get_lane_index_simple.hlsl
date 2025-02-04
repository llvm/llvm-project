// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,CHECK-SPIRV
// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,CHECK-DXIL

// CHECK-SPIRV: define spir_func noundef i32 @{{.*test_1.*}}() [[A0:#[0-9]+]] {
// CHECK-DXIL: define noundef i32 @{{.*test_1.*}}() [[A0:#[0-9]+]] {
// CHECK-SPIRV: %[[CI:[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK-SPIRV: call spir_func i32 @__hlsl_wave_get_lane_index() [ "convergencectrl"(token %[[CI]]) ]
// CHECK-DXIL: call i32 @llvm.dx.wave.getlaneindex()
int test_1() {
  return WaveGetLaneIndex();
}

// CHECK-SPIRV: declare spir_func i32 @__hlsl_wave_get_lane_index() [[A1:#[0-9]+]]
// CHECK-DXIL: declare i32 @llvm.dx.wave.getlaneindex() [[A1:#[0-9]+]]

// CHECK-DAG: attributes [[A0]] = { {{.*}}convergent{{.*}} }
// CHECK-DAG: attributes [[A1]] = { {{.*}}convergent{{.*}} }
