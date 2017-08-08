// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

struct a {
  int b __attribute__((no_caller_saved_registers)); // expected-warning {{'no_caller_saved_registers' only applies to function types; type here is 'int'}}
  static void foo(int *a) __attribute__((no_caller_saved_registers)) {}
};

struct a test __attribute__((no_caller_saved_registers)); // expected-warning {{'no_caller_saved_registers' only applies to function types; type here is 'struct a'}}

__attribute__((no_caller_saved_registers(999))) void bar(int *) {} // expected-error {{'no_caller_saved_registers' attribute takes no arguments}}

void __attribute__((no_caller_saved_registers)) foo(int *){}

[[gnu::no_caller_saved_registers]] void foo2(int *) {}

typedef __attribute__((no_caller_saved_registers)) void (*foo3)(int *);

int (*foo4)(double a, __attribute__((no_caller_saved_registers)) float b); // expected-warning {{'no_caller_saved_registers' only applies to function types; type here is 'float'}}

typedef void (*foo5)(int *);

void foo6(){} // expected-note {{previous declaration is here}}

void __attribute__((no_caller_saved_registers)) foo6(); // expected-error {{function declared with 'no_caller_saved_registers' attribute was previously declared without the 'no_caller_saved_registers' attribute}} 

int main(int argc, char **argv) {
  void (*fp)(int *) = foo; // expected-error {{cannot initialize a variable of type 'void (*)(int *)' with an lvalue of type 'void (int *) __attribute__((no_caller_saved_registers))'}} 
  a::foo(&argc);
  foo3 func = foo2;
  func(&argc);
  foo5 __attribute__((no_caller_saved_registers)) func2 = foo2;
  return 0;
}
