// RUN: %clang_cc1 -std=c++11 -fcxx-exceptions -fexceptions -fsyntax-only -verify -triple wasm32 -Wno-unused-value -target-feature +reference-types %s
// RUN: %clang_cc1 -std=c++20 -fcxx-exceptions -fexceptions -fsyntax-only -verify -triple wasm32 -Wno-unused-value -target-feature +reference-types %s

// 
// Note: As WebAssembly references are sizeless types, we don't exhaustively
// test for cases covered by sizeless-1.c and similar tests.

// Using c++11 to test dynamic exception specifications (which are not 
// allowed in c++17).

// Unlike standard sizeless types, reftype globals are supported.
__externref_t r1;
static __externref_t table[0];

#if (_cplusplus == 201103L)
__externref_t func(__externref_t ref)  throw(__externref_t) { // expected-error {{WebAssembly reference type not allowed in exception specification}}
  return ref;
}
#endif

void *ret_void_ptr() {
  throw table;              // expected-error {{cannot throw a WebAssembly reference type}}
  throw r1;                 // expected-error {{cannot throw a WebAssembly reference type}}
  try {}
  catch (__externref_t T) { // expected-error {{cannot catch a WebAssembly reference type}}
    (void)0;
  }

  return table;             // expected-error {{cannot return a WebAssembly table}}
}
