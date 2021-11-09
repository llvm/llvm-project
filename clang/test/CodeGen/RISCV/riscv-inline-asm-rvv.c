// REQUIRES: riscv-registered-target

// RUN: %clang_cc1 -triple riscv32 -target-feature +experimental-v \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-v \
// RUN:     -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test RISC-V V-extension specific inline assembly constraints.
#include <riscv_vector.h>

void test_v_reg() {
  asm volatile(
      "vsetvli x1, x0, e32,m2,tu,mu\n"
      "vadd.vv v1, v2, v3, v0.t"
      :
      :
      : "v1", "x1");
// CHECK-LABEL: define{{.*}} @test_v_reg
// CHECK: "~{v1},~{x1}"
}

vint32m1_t test_vr(vint32m1_t a, vint32m1_t b) {
// CHECK-LABEL: define{{.*}} @test_vr
// CHECK: %0 = tail call <vscale x 2 x i32> asm sideeffect "vadd.vv $0, $1, $2", "=^vr,^vr,^vr"(<vscale x 2 x i32> %a, <vscale x 2 x i32> %b)
  vint32m1_t ret;
  asm volatile ("vadd.vv %0, %1, %2" : "=vr"(ret) : "vr"(a), "vr"(b));
  return ret;
}

vbool1_t test_vm(vbool1_t a, vbool1_t b) {
// CHECK-LABEL: define{{.*}} @test_vm
// CHECK: %0 = tail call <vscale x 64 x i1> asm sideeffect "vmand.mm $0, $1, $2", "=^vm,^vm,^vm"(<vscale x 64 x i1> %a, <vscale x 64 x i1> %b)
  vbool1_t ret;
  asm volatile ("vmand.mm %0, %1, %2" : "=vm"(ret) : "vm"(a), "vm"(b));
  return ret;
}
