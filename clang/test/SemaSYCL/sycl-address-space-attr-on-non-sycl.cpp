// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -fsyntax-only -fsycl-is-host -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

// The SYCL address space attributes are only enabled in SYCL compilations.

#ifndef SYCL_LANGUAGE_VERSION
// expected-warning@#global   {{'clang::sycl_global' attribute ignored}}
// expected-warning@#local    {{'clang::sycl_local' attribute ignored}}
// expected-warning@#private  {{'clang::sycl_private' attribute ignored}}
// expected-warning@#generic  {{'clang::sycl_generic' attribute ignored}}
// expected-warning@#constant {{'clang::sycl_constant' attribute ignored}}
#else
// expected-no-diagnostics
#endif

using global_ptr = int [[clang::sycl_global]] *;     // #global
using local_ptr = int [[clang::sycl_local]] *;       // #local
using private_ptr = int [[clang::sycl_private]] *;   // #private
using generic_ptr = int [[clang::sycl_generic]] *;   // #generic
using constant_ptr = int [[clang::sycl_constant]] *; // #constant
