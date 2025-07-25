// REQUIRES: riscv-registered-target

// RUN: %clang_cc1 -triple riscv32 -target-feature +v \
// RUN:     -mvscale-min=2 -mvscale-max=2 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:     -mvscale-min=2 -mvscale-max=2 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test RISC-V V-extension fixed-length vector inline assembly constraints.
#include <riscv_vector.h>

typedef vbool1_t fixed_bool1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_i32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint8mf2_t fixed_i8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));

fixed_i32m1_t test_vr(fixed_i32m1_t a) {
// CHECK-LABEL: define{{.*}} @test_vr
// CHECK: %0 = tail call <4 x i32> asm sideeffect "vadd.vv $0, $1, $2", "=^vr,^vr,^vr"(<4 x i32> %a, <4 x i32> %a)
  fixed_i32m1_t ret;
  asm volatile ("vadd.vv %0, %1, %2" : "=vr"(ret) : "vr"(a), "vr"(a));
  return ret;
}

fixed_i8mf2_t test_vd(fixed_i8mf2_t a) {
// CHECK-LABEL: define{{.*}} @test_vd
// CHECK: %0 = tail call <8 x i8> asm sideeffect "vadd.vv $0, $1, $2", "=^vd,^vr,^vr"(<8 x i8> %a, <8 x i8> %a)
  fixed_i8mf2_t ret;
  asm volatile ("vadd.vv %0, %1, %2" : "=vd"(ret) : "vr"(a), "vr"(a));
  return ret;
}

fixed_bool1_t test_vm(fixed_bool1_t a) {
// CHECK-LABEL: define{{.*}} @test_vm
// CHECK: %1 = tail call <16 x i8> asm sideeffect "vmand.mm $0, $1, $2", "=^vm,^vm,^vm"(<16 x i8> %a, <16 x i8> %a)
  fixed_bool1_t ret;
  asm volatile ("vmand.mm %0, %1, %2" : "=vm"(ret) : "vm"(a), "vm"(a));
  return ret;
}
