// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++11 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

struct HType {}; // expected-note-re 6 {{candidate constructor {{.*}} not viable: no known conversion from 'DType'}}
struct DType {}; // expected-note-re 6 {{candidate constructor {{.*}} not viable: no known conversion from 'HType'}}
struct HDType {};

template <typename T> __host__ HType overload_h_d(T a) { return HType(); }
// expected-note@-1 2 {{candidate template ignored: could not match 'HType' against 'DType'}}
// expected-note@-2 2 {{candidate template ignored: target attributes do not match}}
template <typename T> __device__ DType overload_h_d(T a) { return DType(); }
// expected-note@-1 2 {{candidate template ignored: could not match 'DType' against 'HType'}}
// expected-note@-2 2 {{candidate template ignored: target attributes do not match}}

// Check explicit instantiation.
template  __device__ __host__ DType overload_h_d(int a); // There's no HD template...
// expected-error@-1 {{explicit instantiation of 'overload_h_d' does not refer to a function template, variable template, member function, member class, or static data member}}
template  __device__ __host__ HType overload_h_d(int a); // There's no HD template...
// expected-error@-1 {{explicit instantiation of 'overload_h_d' does not refer to a function template, variable template, member function, member class, or static data member}}
template  __device__ DType overload_h_d(int a); // OK. instantiates D
template  __host__ HType overload_h_d(int a); // OK. instantiates H

// Check explicit specialization.
template  <> __device__ __host__ DType overload_h_d(long a); // There's no HD template...
// expected-error@-1 {{no function template matches function template specialization 'overload_h_d'}}
template  <> __device__ __host__ HType overload_h_d(long a); // There's no HD template...
// expected-error@-1 {{no function template matches function template specialization 'overload_h_d'}}
template  <> __device__ DType overload_h_d(long a); // OK. instantiates D
template  <> __host__ HType overload_h_d(long a); // OK. instantiates H


// Can't overload HD template with H or D template, though
// non-template functions are OK.
template <typename T> __host__ __device__ HDType overload_hd(T a) { return HDType(); }
// expected-note@-1 {{previous declaration is here}}
// expected-note@-2 2 {{candidate template ignored: could not match 'HDType' against 'HType'}}
template <typename T> __device__ HDType overload_hd(T a);
// expected-error@-1 {{__device__ function 'overload_hd' cannot overload __host__ __device__ function 'overload_hd'}}
__device__ HDType overload_hd(int a); // OK.

// Verify that target attributes are taken into account when we
// explicitly specialize or instantiate function tempaltes.
template <> __host__ HType overload_hd(int a);
// expected-error@-1 {{no function template matches function template specialization 'overload_hd'}}
template __host__ HType overload_hd(long a);
// expected-error@-1 {{explicit instantiation of 'overload_hd' does not refer to a function template, variable template, member function, member class, or static data member}}
__host__ HType overload_hd(int a); // OK

template <typename T> __host__ T overload_h(T a); // expected-note {{previous declaration is here}}
template <typename T> __host__ __device__ T overload_h(T a);
// expected-error@-1 {{__host__ __device__ function 'overload_h' cannot overload __host__ function 'overload_h'}}
template <typename T> __device__ T overload_h(T a); // OK. D can overload H.

template <typename T> __host__ HType overload_h_d2(T a) { return HType(); }
template <typename T> __host__ __device__ HDType overload_h_d2(T a) { return HDType(); }
template <typename T1, typename T2 = int> __device__ DType overload_h_d2(T1 a) { T1 x; T2 y; return DType(); }

// constexpr functions are implicitly HD, but explicit
// instantiation/specialization must use target attributes as written.
template <typename T> constexpr T overload_ce_implicit_hd(T a) { return a+1; }
// expected-note@-1 3 {{candidate template ignored: target attributes do not match}}

// These will not match the template.
template __host__ __device__ int overload_ce_implicit_hd(int a);
// expected-error@-1 {{explicit instantiation of 'overload_ce_implicit_hd' does not refer to a function template, variable template, member function, member class, or static data member}}
template <> __host__ __device__ long overload_ce_implicit_hd(long a);
// expected-error@-1 {{no function template matches function template specialization 'overload_ce_implicit_hd'}}
template <> __host__ __device__ constexpr long overload_ce_implicit_hd(long a);
// expected-error@-1 {{no function template matches function template specialization 'overload_ce_implicit_hd'}}

// These should work, because template matching ignores the implicit
// HD attributes the compiler gives to constexpr functions/templates,
// so 'overload_ce_implicit_hd' template will match __host__ functions
// only.
template __host__ int overload_ce_implicit_hd(int a);
template <> __host__ long overload_ce_implicit_hd(long a);

template float overload_ce_implicit_hd(float a);
template <> float* overload_ce_implicit_hd(float *a);
template <> constexpr double overload_ce_implicit_hd(double a) { return a + 3.0; };

__host__ void hf() {
  overload_hd(13);
  overload_ce_implicit_hd('h');        // Implicitly instantiated
  overload_ce_implicit_hd(1.0f);       // Explicitly instantiated
  overload_ce_implicit_hd(2.0);        // Explicitly specialized

  HType h = overload_h_d(10);
  HType h2i = overload_h_d2<int>(11);
  HType h2ii = overload_h_d2<int>(12);

  // These should be implicitly instantiated from __host__ template returning HType.
  DType d = overload_h_d(20);          // expected-error {{no viable conversion from 'HType' to 'DType'}}
  DType d2i = overload_h_d2<int>(21);  // expected-error {{no viable conversion from 'HType' to 'DType'}}
  DType d2ii = overload_h_d2<int>(22); // expected-error {{no viable conversion from 'HType' to 'DType'}}
}
__device__ void df() {
  overload_hd(23);
  overload_ce_implicit_hd('d');        // Implicitly instantiated
  overload_ce_implicit_hd(1.0f);       // Explicitly instantiated
  overload_ce_implicit_hd(2.0);        // Explicitly specialized

  // These should be implicitly instantiated from __device__ template returning DType.
  HType h = overload_h_d(10);          // expected-error {{no viable conversion from 'DType' to 'HType'}}
  HType h2i = overload_h_d2<int>(11);  // expected-error {{no viable conversion from 'DType' to 'HType'}}
  HType h2ii = overload_h_d2<int>(12); // expected-error {{no viable conversion from 'DType' to 'HType'}}

  DType d = overload_h_d(20);
  DType d2i = overload_h_d2<int>(21);
  DType d2ii = overload_h_d2<int>(22);
}
