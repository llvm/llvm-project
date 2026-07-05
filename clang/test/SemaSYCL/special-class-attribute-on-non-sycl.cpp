// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -x c++ %s

#ifndef __SYCL_DEVICE_ONLY__
// expected-warning@+5 {{'sycl_special_class' attribute ignored}}
#else
// expected-no-diagnostics
#endif

class __attribute__((sycl_special_class)) special_class {
  void __init(){}
};
