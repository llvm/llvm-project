// RUN: %clang_cc1 -std=c++11 -triple armv7-unknown-linux-gnueabi -fsyntax-only -verify %s

struct a {
  int __attribute__((no_caller_saved_registers)) b;                     // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}
  static void foo(int *a) __attribute__((no_caller_saved_registers)) {} // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}
};

struct a test __attribute__((no_caller_saved_registers)); // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}

__attribute__((no_caller_saved_registers(999))) void bar(int *) {} // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}

__attribute__((no_caller_saved_registers)) void foo(int *){} // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}

[[gnu::no_caller_saved_registers]] void foo2(int *) {} // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}

typedef __attribute__((no_caller_saved_registers)) void (*foo3)(int *); // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}

typedef void (*foo5)(int *);

int (*foo4)(double a, __attribute__((no_caller_saved_registers)) float b); // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}

int main(int argc, char **argv) {
  void (*fp)(int *) = foo;
  a::foo(&argc);
  foo3 func = foo2;
  func(&argc);
  foo5 __attribute__((no_caller_saved_registers)) func2 = foo2; // expected-warning {{unknown attribute 'no_caller_saved_registers' ignored}}
  return 0;
}
