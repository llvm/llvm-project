// RUN: %clang_cc1 %s -O3 -emit-llvm -o - | FileCheck %s
//
// PR13214
// No assumption may be made about the order that a frontend emits branch
// targets (basic blocks). However, the backend's basic block layout makes an
// attempt to preserve source order of control flow, and any bias toward source
// order must start with the frontend.
//
// Note that the frontend inverts branches to simplify the condition, so the
// order of a branch instruction's labels cannot be used as a source order bias.

void calla();
void callb();
void callc();

// CHECK: @test1
// CHECK: @calla
// CHECK: @callb
// CHECK: @callc
// CHECK: ret void
void test1(int a) {
  if (a)
    calla();
  else
    callb();
  callc();
}

// CHECK: @test2
// CHECK: @callb
// CHECK: @calla
// CHECK: @callc
// CHECK: ret void
void test2(int a) {
  if (!a)
    callb();
  else
    calla();
  callc();
}
