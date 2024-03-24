// RUN: %clang_cc1 -emit-llvm -triple mips -target-feature +soft-float %s -o - | FileCheck %s --check-prefix=SOFT_FLOAT

// SOFT_FLOAT: call void asm sideeffect "", "r,~{$1}"(float %1)
void read_float(float *p) {
  __asm__("" ::"r"(*p));
}

// SOFT_FLOAT: call void asm sideeffect "", "r,~{$1}"(double %1)
void read_double(double *p) {
  __asm__("" :: "r"(*p));
}
