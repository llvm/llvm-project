// RUN: %clang_cc1 -triple amdgpu12.00-amd-amdhsa -O0 -emit-llvm -o - %s | FileCheck %s

__attribute__((amdgpu_inst_pref_pad_lines(2)))
kernel void pad_two() {}
// CHECK: define{{.*}} amdgpu_kernel void @pad_two() [[PAD2:#[0-9]+]]

__attribute__((amdgpu_inst_pref_pad_lines(0)))
kernel void pad_zero() {}
// CHECK: define{{.*}} amdgpu_kernel void @pad_zero() [[PAD0:#[0-9]+]]

// CHECK-DAG: attributes [[PAD2]] = {{.*}}"amdgpu-inst-pref-pad-lines"="2"
// CHECK-DAG: attributes [[PAD0]] = {{.*}}"amdgpu-inst-pref-pad-lines"="0"
