// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// Check to ensure these are no-ops/don't really do anything.
void name();

void foo() {
#pragma acc declare
  // CHECK-NOT: declare
#pragma acc routine(name) worker
  // CHECK-NOT: routine
}
#pragma acc declare
  // CHECK-NOT: declare
#pragma acc routine(name) worker
  // CHECK-NOT: routine
