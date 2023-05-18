// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -x c++ %s

#ifndef __SYCL_DEVICE_ONLY__
// expected-warning@+7 {{'sycl_kernel' attribute ignored}}
// expected-warning@+8 {{'sycl_kernel' attribute ignored}}
#else
// expected-no-diagnostics
#endif

template <typename T, typename A, int B>
__attribute__((sycl_kernel)) void foo(T P);
template <typename T, typename A, int B>
[[clang::sycl_kernel]] void foo1(T P);
