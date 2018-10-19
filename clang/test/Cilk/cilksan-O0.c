// Verify that proper Cilksan instrumentation is inserted when a Cilk code is
// compiled at -O0.
//
// Thanks to I-Ting Angelina Lee for contributing this test case.
//
// RUN: %clang_cc1 %s -std=c99 -triple x86_64-unknown-linux-gnu -O0 -fcilkplus -fsanitize=cilk -verify -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int a = 0;
int b = 0;
int c = 2;

void addA() {
  // CHECK: define void @addA()
  // CHECK: __csan_func_entry(i64 {{.+}}, i8* {{.+}}, i64 0)
  a = c;
}

void addB() {
  // CHECK: define void @addB()
  // CHECK: __csan_func_entry(i64 {{.+}}, i8* {{.+}}, i64 0)
  b = a;
}

void foo() {
  // CHECK: define void @foo()
  // CHECK: __csan_func_entry(i64 {{.+}}, i8* {{.+}}, i64 1)
  _Cilk_spawn addA();
  addB();
  _Cilk_sync;
}

int main() {
  // CHECK: define i32 @main()
  // CHECK: __csan_func_entry(i64 {{.+}}, i8* {{.+}}, i64 2)
  foo();
  return 0;
}
