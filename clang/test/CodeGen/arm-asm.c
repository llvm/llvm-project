// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple thumb %s -emit-llvm -o - | FileCheck %s
int t1() {
    static float k = 1.0f;
    // CHECK: flds s15
    __asm__ volatile ("flds s15, %[k] \n" :: [k] "Uv" (k) : "s15");
    return 0;
}

// CHECK-LABEL: @even_reg_constraint_Te
int even_reg_constraint_Te(void) {
  int acc = 0;
  // CHECK: vaddv{{.*\^Te}}
  asm("vaddv.s8 %0, Q0"
      : "+Te" (acc));
  return acc;
}

// CHECK-LABEL: @odd_reg_constraint_To
int odd_reg_constraint_To(void) {
  int eacc = 0, oacc = 0;
  // CHECK: vaddlv{{.*\^To}}
  asm("vaddlv.s8 %0, %1, Q0"
      : "+Te" (eacc), "+To" (oacc));
  return oacc;
}
