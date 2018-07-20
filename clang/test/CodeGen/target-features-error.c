// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -S -verify -o -
int __attribute__((target("avx"), always_inline)) foo(int a) {
  return a + 4;
}
int bar() {
  return foo(4); // expected-error {{always_inline function 'foo' requires target feature 'avx', but would be inlined into function 'bar' that is compiled without support for 'avx'}}
}
