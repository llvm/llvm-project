// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +aes -fsyntax-only -Wno-unused-value -verify=expected,both %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +aes -fsyntax-only -Wno-unused-value -mlong-double-128 -verify=quad,both %s

// GH173254
typedef long double v16dl __attribute__ ((vector_size (16)));

void foo(v16dl *a, v16dl *b) {
    __builtin_ia32_aesenc128(*a, *b); // expected-error {{passing 'v16dl' (vector of 1 'long double' value) to parameter of incompatible type '__attribute__((__vector_size__(2 * sizeof(long long)))) long long' (vector of 2 'long long' values)}}
}

// GH63548
typedef long double T __attribute__((vector_size(32)));

T sum(T first, T second) { return first > second; } // expected-error {{returning '__attribute__((__vector_size__(2 * sizeof(__int128)))) __int128' (vector of 2 '__int128' values) from a function with incompatible result type 'T' (vector of 2 'long double' values)}}

typedef double v16d __attribute__((vector_size(16)));
typedef double v32d __attribute__((vector_size(32)));
typedef long long v2ll __attribute__((vector_size(16)));
typedef __int128 v1i128 __attribute__((vector_size(16)));

v16d vector_cast(v16dl a) {
  return (v16d)a; // expected-error {{invalid conversion between vector type 'v16d' (vector of 2 'double' values) and 'v16dl' (vector of 1 'long double' value) of different size}}
}

void size_mismatch(v16dl a, v32d b) {
  (void)(v32d)a; // both-error {{invalid conversion between vector type 'v32d' (vector of 4 'double' values) and 'v16dl' (vector of 1 'long double' value) of different size}}
  (void)(v16dl)b; // both-error {{invalid conversion between vector type 'v16dl' (vector of 1 'long double' value) and 'v32d' (vector of 4 'double' values) of different size}}
}

__int128 integer_cast(v16dl a) {
  return (__int128)a; // expected-error {{invalid conversion between vector type 'v16dl' (vector of 1 'long double' value) and integer type '__int128' of different size}}
}

v2ll scalar_operand(v2ll a, long double b) {
  return a + b; // expected-error {{cannot convert between scalar type 'long double' and vector type 'v2ll' (vector of 2 'long long' values) as implicit conversion would cause truncation}}
}

long double scalar_result(v1i128 a) {
  return a; // expected-error {{returning 'v1i128' (vector of 1 '__int128' value) from a function with incompatible result type 'long double'}}
}

long double same_element(v16dl a) {
  return a;
}

void foo_double(v16d *a, v16d *b) {
    __builtin_ia32_aesenc128(*a, *b);
}

v32d sum_double(v32d first, v32d second) { return first > second; }
