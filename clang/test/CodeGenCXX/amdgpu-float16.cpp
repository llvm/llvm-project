// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx701 -S -o - %s | FileCheck %s -check-prefix=NOF16
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx803 -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -S -o - %s | FileCheck %s
void f() {
  _Float16 x, y, z;
  // CHECK: v_add_f16_e64
  // NOF16: v_add_f32_e64
  z = x + y;
  // CHECK: v_sub_f16_e64
  // NOF16: v_sub_f32_e64
  z = x - y;
  // CHECK: v_mul_f16_e64
  // NOF16: v_mul_f32_e64
  z = x * y;
  // CHECK: v_div_fixup_f16
  // NOF16: v_div_fixup_f32
  z = x / y;
}
