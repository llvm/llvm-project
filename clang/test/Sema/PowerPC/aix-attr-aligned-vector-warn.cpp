// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-feature +altivec -target-cpu pwr7 -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec -target-cpu pwr7 -verify -fsyntax-only %s

struct alignas(8) Align8 {
  void *a, *b;
};

alignas(8) vector int V1; // expected-warning {{requested alignment is less than minimum alignment of 16 for type '__vector int' (vector of 4 'int' values)}}
alignas(Align8) vector int V2; // expected-warning {{requested alignment is less than minimum alignment of 16 for type '__vector int' (vector of 4 'int' values)}}
