// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -fexceptions -fms-extensions -verify %s

// test for issue (https://github.com/llvm/llvm-project/issues/222931)

// Reject throwing/catching pointers with non-default address spaces in Sema because
// runtimes dont yet support for cross-address-space pointer conversions

void test_throw() {
  throw (int * __ptr32)0; // expected-error {{cannot throw pointer with address space}}
  throw (int __attribute__((address_space(1))) *)0; // expected-error {{cannot throw pointer with address space}}
}

void test_catch() {
  try {
  } catch (int * __ptr32 p) { // expected-error {{cannot catch pointer with address space}}
  } catch (int __attribute__((address_space(1))) * p) { // expected-error {{cannot catch pointer with address space}}
  }
}
