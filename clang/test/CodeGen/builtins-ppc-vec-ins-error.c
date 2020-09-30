// REQUIRES: powerpc-registered-target

// RUN: %clang_cc1 -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64le-unknown-unknown -fsyntax-only %s -verify
// RUN: %clang_cc1 -target-feature +vsx -target-cpu pwr10 \
// RUN:   -triple powerpc64-unknown-unknown -fsyntax-only %s -verify

#include <altivec.h>

vector signed int vsia;
vector unsigned int vuia;
vector signed long long vslla;
vector unsigned long long vulla;
vector float vfa;
vector double vda;
signed int sia;
unsigned int uia;
signed long long slla;
unsigned long long ulla;
float fa;
double da;

vector signed int test_vec_replace_elt_si(void) {
  return vec_replace_elt(vsia, sia, 13); // expected-error {{argument value 13 is outside the valid range [0, 12]}}
}

vector unsigned int test_vec_replace_elt_ui(void) {
  return vec_replace_elt(vuia, sia, 1); // expected-error {{arguments are of different types ('unsigned int' vs 'int')}}
}

vector float test_vec_replace_elt_f(void) {
  return vec_replace_elt(vfa, fa, 20); // expected-error {{argument value 20 is outside the valid range [0, 12]}}
}

vector float test_vec_replace_elt_f_2(void) {
  return vec_replace_elt(vfa, da, 0); // expected-error {{arguments are of different types ('float' vs 'double')}}
}

vector signed long long test_vec_replace_elt_sll(void) {
  return vec_replace_elt(vslla, slla, 9); // expected-error {{argument value 9 is outside the valid range [0, 8]}}
}

vector unsigned long long test_vec_replace_elt_ull(void) {
  return vec_replace_elt(vulla, vda, 0); // expected-error {{arguments are of different types ('unsigned long long' vs '__vector double' (vector of 2 'double' values))}}
}

vector unsigned long long test_vec_replace_elt_ull_2(void) {
  return vec_replace_elt(vulla, vulla, vsia); // expected-error {{argument to '__builtin_altivec_vec_replace_elt' must be a constant integer}}
}

vector double test_vec_replace_elt_d(void) {
  return vec_replace_elt(vda, da, 33); // expected-error {{argument value 33 is outside the valid range [0, 8]}}
}

vector unsigned char test_vec_replace_unaligned_si(void) {
  return vec_replace_unaligned(vsia, da, 6); // expected-error {{arguments are of different types ('int' vs 'double')}}
}

vector unsigned char test_vec_replace_unaligned_ui(void) {
  return vec_replace_unaligned(vuia, uia, 14); // expected-error {{argument value 14 is outside the valid range [0, 12]}}
}

vector unsigned char test_vec_replace_unaligned_f(void) {
  return vec_replace_unaligned(vfa, fa, 19); // expected-error {{argument value 19 is outside the valid range [0, 12]}}
}

vector unsigned char test_vec_replace_unaligned_sll(void) {
  return vec_replace_unaligned(vslla, fa, 0); // expected-error {{arguments are of different types ('long long' vs 'float')}}
}

vector unsigned char test_vec_replace_unaligned_ull(void) {
  return vec_replace_unaligned(vulla, ulla, 12); // expected-error {{argument value 12 is outside the valid range [0, 8]}}
}

vector unsigned char test_vec_replace_unaligned_d(void) {
  return vec_replace_unaligned(vda, fa, 8); // expected-error {{arguments are of different types ('double' vs 'float')}}
}

vector unsigned char test_vec_replace_unaligned_d_2(void) {
  return vec_replace_unaligned(vda, vda, da); // expected-error {{argument to '__builtin_altivec_vec_replace_unaligned' must be a constant integer}}
}
