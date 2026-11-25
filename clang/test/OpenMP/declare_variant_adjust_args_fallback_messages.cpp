// RUN: %clang_cc1 -fopenmp -fopenmp-version=61 -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=61 -std=c++14 -fsyntax-only -verify %s

void variant(int *A, int *B);

// Fallback modifiers are only allowed on need_device_ptr.
#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(nothing(fb_nullify): A) // expected-error {{unexpected modifier in OpenMP clause 'adjust_args'}}
void nothing_nullify(int *A, int *B);

#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(nothing(fb_preserve): 1) // expected-error {{unexpected modifier in OpenMP clause 'adjust_args'}}
void nothing_preserve(int *A, int *B);

#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(need_device_addr(fb_nullify): 1:2) // expected-error {{unexpected modifier in OpenMP clause 'adjust_args'}}
void addr_nullify(int *A, int *B);

#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(need_device_addr(fb_preserve): A) // expected-error {{unexpected modifier in OpenMP clause 'adjust_args'}}
void addr_preserve(int *A, int *B);

// Both fallback modifiers remain valid on need_device_ptr.
#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(need_device_ptr(fb_nullify): A, B)
void ptr_nullify(int *A, int *B);

#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(need_device_ptr(fb_preserve): 1:2)
void ptr_preserve(int *A, int *B);

// The other operations remain valid without a fallback modifier.
#pragma omp declare variant(variant) match(construct={dispatch}) \
  adjust_args(nothing: 1:2)
void nothing_plain(int *A, int *B);

void reference_variant(int &A, int &B);

#pragma omp declare variant(reference_variant) match(construct={dispatch}) \
  adjust_args(need_device_addr: 1:2)
void addr_plain(int &A, int &B);
