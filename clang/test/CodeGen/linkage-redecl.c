// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK: @test2_i = internal global i32 99
static int test2_i = 99;
int test2_f(void) {
  extern int test2_i;
  return test2_i;
}

// C99 6.2.2p3
// PR3425
static void f(int x);

void g0(void) {
  f(5);
}

extern void f(int x) { } // still has internal linkage
// CHECK-LABEL: define internal {{.*}}void @f
