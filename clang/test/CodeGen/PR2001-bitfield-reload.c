// RUN: %clang_cc1 -triple i386-unknown-unknown -O3 -emit-llvm -o - %s | FileCheck %s
// PR2001

/* Test that the result of the assignment properly uses the value *in
   the bitfield* as opposed to the RHS. */
static int foo(int i) {
  struct {
    int f0 : 2;
  } x;
  return (x.f0 = i);
}

int bar(void) {
  // CHECK: ret i32 1
  return foo(-5) == -1;
}
