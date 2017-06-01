// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -o - -verify -fsyntax-only

typedef __attribute__(( ext_vector_type(3) )) char char3;
typedef __attribute__(( ext_vector_type(16) )) char char16;

char3 f1(char16 x) {
  return  __builtin_astype(x, char3); // expected-error{{invalid reinterpretation: sizes of 'char3' (vector of 3 'char' values) and 'char16' (vector of 16 'char' values) must match}}
}

char16 f3(int x) {
  return __builtin_astype(x, char16); // expected-error{{invalid reinterpretation: sizes of 'char16' (vector of 16 'char' values) and 'int' must match}}
}

