// RUN: %clang_cc1 %s -O0 -disable-O0-optnone -emit-llvm -fextend-lifetimes -fexceptions -o - | FileCheck %s --implicit-check-not="landingpad {"

// Check that fake uses do not mistakenly cause a landing pad to be generated when
// exceptions are enabled.

extern void bar(int);
void foo(int p) {
  int a = 17;
  bar(a);
}

// CHECK:      define {{.*}} @foo
// CHECK-NOT:  personality
// CHECK:      call void (...) @llvm.fake.use
