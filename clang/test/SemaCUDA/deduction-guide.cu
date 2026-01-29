// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device -verify=expected,dev %s
// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -verify %s

#include "Inputs/cuda.h"

template <class T>
struct CTADType { // expected-note 2{{candidate constructor (the implicit copy constructor) not viable: requires 1 argument, but 3 were provided}}
                  // expected-note@-1 2{{candidate constructor (the implicit move constructor) not viable: requires 1 argument, but 3 were provided}}
  T first;
  T second;

  CTADType(T x) : first(x), second(x) {} // expected-note 2{{candidate constructor not viable: requires single argument 'x', but 3 arguments were provided}}
  __device__ CTADType(T x) : first(x), second(x) {} // expected-note 2{{candidate constructor not viable: requires single argument 'x', but 3 arguments were provided}}
  __host__ __device__ CTADType(T x, T y) : first(x), second(y) {} // expected-note 2{{candidate constructor not viable: requires 2 arguments, but 3 were provided}}
  CTADType(T x, T y, T z) : first(x), second(z) {} // dev-note {{'CTADType' declared here}}
                                                   // expected-note@-1 {{candidate constructor not viable: call to __host__ function from __device__ function}}
                                                   // expected-note@-2 {{candidate constructor not viable: call to __host__ function from __global__ function}}
};

template <class T>
CTADType(T, T) -> CTADType<T>;

__host__ __device__ void use_ctad_host_device() {
  CTADType ctad_from_two_args(1, 1);
  CTADType ctad_from_one_arg(1);
  CTADType ctad_from_three_args(1, 2, 3); // dev-error {{reference to __host__ function 'CTADType' in __host__ __device__ function}}
}

__host__ void use_ctad_host() {
  CTADType ctad_from_two_args(1, 1);
  CTADType ctad_from_one_arg(1);
  CTADType ctad_from_three_args(1, 2, 3);
}

__device__ void use_ctad_device() {
  CTADType ctad_from_two_args(1, 1);
  CTADType ctad_from_one_arg(1);
  CTADType<int> ctad_from_three_args(1, 2, 3); // expected-error {{no matching constructor for initialization of 'CTADType<int>'}}
}

__global__ void use_ctad_global() {
  CTADType ctad_from_two_args(1, 1);
  CTADType ctad_from_one_arg(1);
  CTADType<int> ctad_from_three_args(1, 2, 3); // expected-error {{no matching constructor for initialization of 'CTADType<int>'}}
}
