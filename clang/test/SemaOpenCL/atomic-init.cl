// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -fsyntax-only -verify  %s

global atomic_int a1 = 0;

kernel void test_atomic_initialization() {
  a1 = 1; // expected-error {{atomic variable can be assigned to a variable only in global address space}}
  atomic_int a2 = 0; // expected-error {{atomic variable can be initialized to a variable only in global address space}}
  private atomic_int a3 = 0; // expected-error {{atomic variable can be initialized to a variable only in global address space}}
  local atomic_int a4 = 0; // expected-error {{'__local' variable cannot have an initializer}}
  global atomic_int a5 = 0; // expected-error {{function scope variable cannot be declared in global address space}}
  static global atomic_int a6 = 0;
}
