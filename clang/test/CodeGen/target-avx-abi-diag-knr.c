// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -std=c17 -verify=no256,knr -o - -S
// RUN: %clang_cc1 %s -triple=x86_64-linux-gnu -std=c17 -target-feature +avx -verify=knr -o - -S
// REQUIRES: x86-registered-target

typedef short avx256Type __attribute__((vector_size(32)));

// knr-warning@+3 {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
// no256-warning@+2 {{AVX vector return of type 'avx256Type' (vector of 16 'short' values) without 'avx' enabled changes the ABI}}
// no256-warning@+2 {{AVX vector argument of type 'avx256Type' (vector of 16 'short' values) without 'avx' enabled changes the ABI}}
avx256Type knr_def(x)
  avx256Type x;
{
  return x;
}
