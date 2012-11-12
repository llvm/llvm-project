// RUN: %clang_cc1 -triple i686-win32 -emit-llvm -fms-extensions < %s | FileCheck %s

void bar() {
}

// CHECK-NOT: foo
__forceinline void foo() {
  bar();
}

void i_want_bar() {
// CHECK: call void @bar
  foo();
}
