// RUN: %clang_cc1 -triple thumbv7m-apple-darwin-eabi -ast-dump %s | FileCheck %s

// CHECK: CallExpr {{.*}} 'int'

void foo(int a, int *b) {
  do {
  } while (__builtin_arm_strex(a, b));
}
