// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
int __attribute__((target("avx"), always_inline)) foo(int a) {
  return a + 4;
}
int bar() {
  return foo(4); // expected-error {{function 'bar' and always_inline callee function 'foo' are required to have matching target features}}
}

