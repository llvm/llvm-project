// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -triple=x86_64-unknown-linux-gnu -fsyntax-only -verify

#include <next32_scratchpad.h>
#include <stdint.h>

// Keyword thread_local is a part of C++11 standard, equivalent to __thread.
__next32_global__ thread_local int32_t arr1[34];   // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_constant__ thread_local int32_t arr2[34]; // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
__next32_local__ thread_local int32_t arr3[34]; // expected-error{{invalid usage of address space, only tls address space can be used with __thread/_Thread_local/thread_local}}
