// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions -fms-extensions -verify %s

// Test for issue (https://github.com/llvm/llvm-project/issues/222931).
// Reject throwing/catching pointers and references with non-default address
// spaces because runtimes don't yet support cross-address-space conversions.

using as1_int = int __attribute__((address_space(1)));
using as1_int_ptr = int * __ptr32;

void test_throw() {
  throw (int * __ptr32)0; // expected-error {{cannot throw pointer with non-default address space}}
  throw (as1_int *)0;     // expected-error {{cannot throw pointer with non-default address space}}
  throw (int *)0;         // ok
}

void test_catch() {
  try {
  } catch (int * __ptr32 p) { // expected-error {{cannot catch pointer with non-default address space}}
  }
  try {
  } catch (as1_int *p) {      // expected-error {{cannot catch pointer with non-default address space}}
  }
  try {
  } catch (as1_int &p) {      // expected-error {{cannot catch reference with non-default address space}}
  }
  try {
  } catch (as1_int &&p) {     // expected-error {{cannot catch exceptions by rvalue reference}}
                              // expected-error@-1 {{cannot catch reference with non-default address space}}
  }
  try {
  } catch (as1_int_ptr &p) {  // expected-error {{cannot catch reference with non-default address space}}
  }
  try {
  } catch (int &p) {          // ok
  }
  try {
  } catch (int *&p) {         // ok
  }
try {  
} catch (as1_int p) {         // expected-error {{cannot catch pointer with non-default address space}}  
}
}
