// RUN: %clang_cc1 %s -O3 -emit-llvm -fextend-lifetimes -fexceptions -o - | FileCheck %s

// Check that fake uses do not mistakenly cause a landing pad to be generated when
// exceptions are enabled.

extern void bar(int);
void foo(int p) {
  int a = 17;
  bar(a);
}

// CHECK:      define {{.*}} @foo
// CHECK-NOT:  personality
// CHECK:      entry:
// CHECK:      llvm.fake.use
// CHECK-NOT:  landingpad
