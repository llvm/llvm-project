// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-feature +altivec -target-cpu pwr7 -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec -target-cpu pwr7 -verify -fsyntax-only %s

int escape(vector int*);

typedef vector int __attribute__((aligned(8))) UnderAlignedVI;
UnderAlignedVI TypedefedGlobal;

vector int V __attribute__((aligned(8))); // expected-warning {{requested alignment is less than minimum alignment of 16 for type '__vector int' (vector of 4 'int' values)}}
vector int V __attribute__((aligned(8))); // expected-warning {{requested alignment is less than minimum alignment of 16 for type '__vector int' (vector of 4 'int' values)}}

int localTypedefed(void) {
  UnderAlignedVI TypedefedLocal;
  return escape(&TypedefedLocal); // expected-warning {{passing 8-byte aligned argument to 16-byte aligned parameter 1 of 'escape' may result in an unaligned pointer access}}
}
