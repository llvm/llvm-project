// REQUIRES: riscv-registered-target

// RUN: %clang_cc1 -triple riscv32 -target-feature +v \
// RUN:     -mvscale-min=2 -mvscale-max=2 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:     -mvscale-min=2 -mvscale-max=2 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test RISC-V V-extension fixed-length vector inline assembly constraints.
#include <riscv_vector.h>
#include <stdbool.h>

typedef vbool1_t fixed_bool1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_i32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint8mf2_t fixed_i8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2)));

typedef bool bx2 __attribute__((ext_vector_type(16)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef char  i8x4 __attribute__((ext_vector_type(4)));

fixed_i32m1_t test_vr(fixed_i32m1_t a) {
// CHECK-LABEL: define{{.*}} @test_vr
// CHECK: %0 = tail call <4 x i32> asm sideeffect "vadd.vv $0, $1, $2", "=^vr,^vr,^vr"(<4 x i32> %a, <4 x i32> %a)
  fixed_i32m1_t ret;
  asm volatile ("vadd.vv %0, %1, %2" : "=vr"(ret) : "vr"(a), "vr"(a));
  return ret;
}

i32x2 test_vr2(i32x2 a) {
// CHECK-LABEL: define{{.*}} @test_vr2
// CHECK: %1 = tail call <2 x i32> asm sideeffect "vadd.vv $0, $1, $2", "=^vr,^vr,^vr"(<2 x i32> %0, <2 x i32> %0)
  i32x2 ret;
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

i8x4 test_vd2(i8x4 a) {
// CHECK-LABEL: define{{.*}} @test_vd2
// CHECK: %1 = tail call <4 x i8> asm sideeffect "vadd.vv $0, $1, $2", "=^vd,^vr,^vr"(<4 x i8> %0, <4 x i8> %0)
  i8x4 ret;
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

void test_vm2(bx2 a) {
// CHECK-LABEL: define{{.*}} @test_vm2
// CHECK: tail call void asm sideeffect "dummy $0", "^vm"(<16 x i1> %a1)
  asm volatile ("dummy %0" :: "vm"(a));
}
