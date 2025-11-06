// RUN: %clang_cc1 -verify -triple x86_64-unknown-unknown -emit-llvm-only %s

// Test that MMX register constraint 'y' with mismatched vector sizes
// produces a proper error message instead of an assertion failure.

typedef int vec256 __attribute__((ext_vector_type(8)));

vec256 foo(vec256 in) {
  vec256 out;

  asm("something %0" : : "y"(in)); // expected-error {{invalid input size for constraint 'y'}}
  asm("something %0" : "=y"(out)); // expected-error {{invalid output size for constraint '=y'}}
  asm("something %0, %0" : "+y"(out)); // expected-error {{invalid output size for constraint '+y'}}

  return out;
}

// Additional tests for different vector sizes
typedef int vec128 __attribute__((ext_vector_type(4)));
typedef int vec64  __attribute__((ext_vector_type(2)));

void test_128bit_mismatch() {
  vec128 out;
  __asm__("nop" : "=y"(out)); // expected-error {{invalid output size for constraint '=y'}}
}

void test_64bit_valid() {
  // This should work - 64-bit vector matches MMX register size
  vec64 out;
  __asm__("nop" : "=y"(out));
}
