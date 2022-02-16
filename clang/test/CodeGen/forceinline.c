// RUN: %clang_cc1 -triple i686-win32 -emit-llvm -fms-extensions < %s | FileCheck %s

void bar(void) {
}

// CHECK-NOT: foo
__forceinline void foo(void) {
  bar();
}

void i_want_bar(void) {
// CHECK: call void @bar
  foo();
}
