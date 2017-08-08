// Test this without pch.
// RUN: %clang_cc1 %s -DHEADER -DHEADER_USER -triple spir-unknown-unknown -verify -pedantic -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 %s -DHEADER -triple spir-unknown-unknown -emit-pch -o %t -verify -pedantic
// RUN: %clang_cc1 %s -DHEADER_USER -triple spir-unknown-unknown -include-pch %t -fsyntax-only -verify -pedantic

#if defined(HEADER) && !defined(INCLUDED)
#define INCLUDED 

#pragma OPENCL EXTENSION all : begin // expected-warning {{expected 'disable' - ignoring}}
#pragma OPENCL EXTENSION all : end // expected-warning {{expected 'disable' - ignoring}}

#pragma OPENCL EXTENSION my_ext : begin 

struct A {
  int a;
};

typedef struct A TypedefOfA;
typedef const TypedefOfA* PointerOfA;

void f(void);

__attribute__((overloadable)) void g(long x);

#pragma OPENCL EXTENSION my_ext : end
#pragma OPENCL EXTENSION my_ext : end // expected-warning {{OpenCL extension end directive mismatches begin directive - ignoring}}

__attribute__((overloadable)) void g(void);

#endif // defined(HEADER) && !defined(INCLUDED)

#ifdef HEADER_USER

#pragma OPENCL EXTENSION my_ext : enable
void test_f1(void) {
  struct A test_A1;
  f();
  g(0);
}

#pragma OPENCL EXTENSION my_ext : disable 
void test_f2(void) {
  struct A test_A2; // expected-error {{use of type 'struct A' requires my_ext extension to be enabled}}
  const struct A test_A_local; // expected-error {{use of type 'struct A' requires my_ext extension to be enabled}}
  TypedefOfA test_typedef_A; // expected-error {{use of type 'TypedefOfA' (aka 'struct A') requires my_ext extension to be enabled}}
  PointerOfA test_A_pointer; // expected-error {{use of type 'PointerOfA' (aka 'const struct A *') requires my_ext extension to be enabled}}
  f(); // expected-error {{use of declaration 'f' requires my_ext extension to be enabled}}
  g(0); // expected-error {{no matching function for call to 'g'}}
        // expected-note@-26 {{candidate disabled due to OpenCL extension}}
        // expected-note@-22 {{candidate function not viable: requires 0 arguments, but 1 was provided}}
}

#endif // HEADER_USER

