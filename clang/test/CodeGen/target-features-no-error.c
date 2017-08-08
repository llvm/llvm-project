// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -emit-llvm -o - -target-feature -sse2

// Verify that negative features don't cause additional requirements on the inline function.
int __attribute__((target("sse"), always_inline)) foo(int a) {
  return a + 4;
}
int bar() {
  return foo(4); // expected-no-diagnostics
}
