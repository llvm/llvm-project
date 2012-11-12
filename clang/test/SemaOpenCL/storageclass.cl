// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2

static int A;

// static is not allowed at local scope.
void kernel foo() {
  static int X = 5; // expected-error{{variables in function scope cannot be declared static}} 
  auto int Y = 7; // expected-error{{OpenCL does not support the 'auto' storage class specifier}}
}

static void kernel bar() { // expected-error{{kernel functions cannot be declared static}}
}
