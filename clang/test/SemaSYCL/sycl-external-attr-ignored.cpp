// RUN: %clang_cc1 -fsyntax-only -verify %s

// These tests validate that the sycl_external attribute is ignored when SYCL
// support is not enabled.

// expected-warning@+1{{'clang::sycl_external' attribute ignored}}
[[clang::sycl_external]] void bar() {}

// expected-warning@+1{{'clang::sycl_external' attribute ignored}}
[[clang::sycl_external]] int a;

// expected-warning@+2{{'clang::sycl_external' attribute ignored}}
template<typename T>
[[clang::sycl_external]] void ft(T) {}
template void ft(int);
