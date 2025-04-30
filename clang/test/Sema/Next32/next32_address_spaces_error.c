// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple=x86_64-unknown-linux-gnu -fsyntax-only -verify

#include <next32_scratchpad.h>
#include <stdint.h>

__next32_tls__ int32_t arr1[34];               // expected-error{{invalid usage of tls address space, this address space must be used with __thread/_Thread_local/thread_local}}
__next32_global__ __thread int32_t arr2[34];   // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_constant__ __thread int32_t arr3[34]; // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_local__ __thread int32_t arr4[34];    // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_global__ _Thread_local int32_t arr5[34];   // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_constant__ _Thread_local int32_t arr6[34]; // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_local__ _Thread_local int32_t arr7[34]; // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
