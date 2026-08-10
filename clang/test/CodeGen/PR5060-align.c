// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK: @foo.p = internal global i8 0, align 16
char *foo(void) {
  static char p __attribute__((aligned(16)));
  return &p;
}

void bar(long n) {
  // CHECK: align 16
  char p[n] __attribute__((aligned(16)));
}

