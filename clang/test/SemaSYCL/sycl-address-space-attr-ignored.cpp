// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -fsyntax-only -verify -x clcpp %s
// RUN: %clang_cc1 -fsyntax-only -verify -x cuda %s
// RUN: %clang_cc1 -fsyntax-only -verify -x hip %s

// The SYCL address space attributes are only enabled in SYCL compilations.

// expected-warning@+1 {{'clang::sycl_global' attribute ignored}}
using global_ptr = int [[clang::sycl_global]] *;

// expected-warning@+1 {{'clang::sycl_local' attribute ignored}}
using local_ptr = int [[clang::sycl_local]] *;

// expected-warning@+1 {{'clang::sycl_private' attribute ignored}}
using private_ptr = int [[clang::sycl_private]] *;

// expected-warning@+1 {{'clang::sycl_generic' attribute ignored}}
using generic_ptr = int [[clang::sycl_generic]] *;

// expected-warning@+1 {{'clang::sycl_constant' attribute ignored}}
using constant_ptr = int [[clang::sycl_constant]] *;
