// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=clc++ -verify -fsyntax-only -fexceptions -fcxx-exceptions

// This test checks that various C++ and OpenCL C keywords are not available
// in OpenCL.

// Test that exceptions are disabled despite passing -fcxx-exceptions.
kernel void test_exceptions() {
  int x;
  try {
    // expected-error@-1 {{cannot use 'try' with exceptions disabled}}
    throw 0;
    // expected-error@-1 {{cannot use 'throw' with exceptions disabled}}
  } catch (int i) {
    x = 41;
  }
}

// Test that only __-prefixed address space qualifiers are accepted.
struct test_address_space_qualifiers {
  global int *g;
  __global int *uug;
  int global; // expected-warning{{declaration does not declare anything}}

  local int *l;
  __local int *uul;
  int local; // expected-warning{{declaration does not declare anything}}

  private int *p;
  __private int *uup;
  int private; // expected-warning{{declaration does not declare anything}}

  constant int *c;
  __constant int *uuc;
  int constant; // expected-warning{{declaration does not declare anything}}

  generic int *ge;
  __generic int *uuge;
  int generic; // expected-warning{{declaration does not declare anything}}
};

// Test that 'private' can be parsed as an access qualifier and an address space too.
class A{
  private:
  private int i; //expected-error{{field may not be qualified with an address space}}
};

private ::A i; //expected-error{{program scope variable must reside in global or constant address space}}

void foo(private int i);

private int bar(); //expected-error{{return value cannot be qualified with address space}}
