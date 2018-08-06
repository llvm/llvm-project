// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -verify -fsyntax-only
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -verify -fsyntax-only -fexceptions -fcxx-exceptions

// This test checks that various C++ and OpenCL C keywords are not available
// in OpenCL C++, according to OpenCL C++ 1.0 Specification Section 2.9.

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
  // expected-error@-1 {{unknown type name 'global'}}
  // expected-error@-2 {{expected member name or ';' after declaration specifiers}}
  __global int *uug;
  int global; // should be fine in OpenCL C++

  local int *l;
  // expected-error@-1 {{unknown type name 'local'}}
  // expected-error@-2 {{expected member name or ';' after declaration specifiers}}
  __local int *uul;
  int local; // should be fine in OpenCL C++

  private int *p;
  // expected-error@-1 {{expected ':'}}
  __private int *uup;
  int private; // 'private' is a keyword in C++14 and thus in OpenCL C++
  // expected-error@-1 {{expected member name or ';' after declaration specifiers}}

  constant int *c;
  // expected-error@-1 {{unknown type name 'constant'}}
  // expected-error@-2 {{expected member name or ';' after declaration specifiers}}
  __constant int *uuc;
  int constant; // should be fine in OpenCL C++

  generic int *ge;
  // expected-error@-1 {{unknown type name 'generic'}}
  // expected-error@-2 {{expected member name or ';' after declaration specifiers}}
  __generic int *uuge;
  int generic; // should be fine in OpenCL C++
};
