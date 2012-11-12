// RUN: %clang_cc1 -emit-llvm %s -o - -verify | FileCheck %s

// CHECK: @foo.p = internal global i8 0, align 32
char *foo(void) {
  static char p __attribute__((aligned(32)));
  return &p;
}

void bar(long n) {
  // CHECK: align 32
  char p[n] __attribute__((aligned(32)));
}

