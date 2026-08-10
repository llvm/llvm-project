// REQUIRES: hexagon-registered-target

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf \
// RUN:   -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -O0 -internal-isystem %S/../../lib/Headers/ \
// RUN:   -target-cpu hexagonv68 -triple hexagon-unknown-elf -x c++ \
// RUN:   -emit-llvm %s -o - | FileCheck %s

#include <hexagon_protos.h>

// expected-no-diagnostics

void test_protos(float a, unsigned int b) {
  unsigned char c;
  // CHECK: call i64 @llvm.hexagon.A2.absp
  b = Q6_P_abs_P(b);
}

void test_dma() {
  unsigned int b;

  // CHECK: call i32 @llvm.hexagon.Y6.dmpoll
  b = Q6_R_dmpoll();
  // CHECK: call i32 @llvm.hexagon.Y6.dmpause
  b = Q6_R_dmpause();
}
