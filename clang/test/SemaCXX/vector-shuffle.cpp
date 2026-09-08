// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify

typedef bool v8b __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

void vector_of_bool_mask() {
  v8b a;
  v8b b;
  auto r = __builtin_shufflevector(a, b); // expected-error {{second argument to __builtin_shufflevector must be a vector of any integer type other than boolean (was 'v8b' (vector of 8 'bool' values))}}
}

void vector_of_float_mask() {
  v8f a;
  v8f b;
  auto r = __builtin_shufflevector(a, b); // expected-error {{second argument to __builtin_shufflevector must be a vector of any integer type other than boolean (was 'v8f' (vector of 8 'float' values))}}
}
