// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -std=c++20 -target-feature +power9-vector -target-feature +isa-v207-instructions \
// RUN:   -triple powerpc64le-unknown-unknown -fsyntax-only -verify %s

// expected-no-diagnostics
#include <altivec.h>

auto test_u64(vector unsigned long long a, vector unsigned long long b, vector unsigned __int128 c) {
  return vec_msum(a, b, c);
}

auto test_direct(vector unsigned long long a, vector unsigned long long b, vector unsigned __int128 c) {
  return vec_vmsumudm(a, b, c);
}
