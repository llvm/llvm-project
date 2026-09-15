// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -std=c++20 -target-feature +power9-vector -target-feature +isa-v207-instructions \
// RUN:   -triple powerpc64le-unknown-unknown -fsyntax-only -flax-vector-conversions=none \
// RUN:   -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -std=c++20 -target-feature +power9-vector -target-feature +isa-v207-instructions \
// RUN:   -triple powerpc64-unknown-unknown -fsyntax-only -flax-vector-conversions=none \
// RUN:   -verify -verify-ignore-unexpected=note %s

#include <altivec.h>

auto test_u64(vector unsigned long long a, vector unsigned long long b, vector unsigned __int128 c) {
  return vec_msum(a, b, c);
}

auto test_direct(vector unsigned long long a, vector unsigned long long b, vector unsigned __int128 c) {
  return vec_vmsumudm(a, b, c);
}

void test_error(vector signed long long a, vector signed long long b, vector signed __int128 c,
                vector unsigned long long u, vector unsigned __int128 uc) {
  vec_msum(a, b, c); // expected-error {{no matching function for call to 'vec_msum'}}
  vec_msum(u, u, c); // expected-error {{no matching function for call to 'vec_msum'}}
  vec_vmsumudm(a, b, uc); // expected-error {{no matching function for call to 'vec_vmsumudm'}}
  vec_vmsumudm(u, u, c); // expected-error {{no matching function for call to 'vec_vmsumudm'}}
}
