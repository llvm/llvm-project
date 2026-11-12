// RUN: %clang_cc1 %s -verify
// expected-no-diagnostics

constant char * __constant x = "hello world";

void foo(__constant char * a) {

}

void bar() {
  foo("hello world");
  foo(x);
}
