// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: @use_flat_scratch_name
kernel void use_flat_scratch_name()
{
// CHECK: tail call{{.*}} void asm sideeffect "s_mov_b64 flat_scratch, 0", "~{flat_scratch}"()
  __asm__ volatile("s_mov_b64 flat_scratch, 0" : : : "flat_scratch");

// CHECK: tail call{{.*}} void asm sideeffect "s_mov_b32 flat_scratch_lo, 0", "~{flat_scratch_lo}"()
  __asm__ volatile("s_mov_b32 flat_scratch_lo, 0" : : : "flat_scratch_lo");

// CHECK: tail call{{.*}} void asm sideeffect "s_mov_b32 flat_scratch_hi, 0", "~{flat_scratch_hi}"()
  __asm__ volatile("s_mov_b32 flat_scratch_hi, 0" : : : "flat_scratch_hi");
}
