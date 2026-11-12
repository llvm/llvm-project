// RUN: %clang_cc1 -fsyntax-only -verify=notsycl -Wno-unused %s
// RUN: %clang_cc1 -fsyntax-only -fsycl-is-host -verify=sycl -Wno-unused %s
// RUN: %clang_cc1 -fsyntax-only -fsycl-is-device -verify=sycl -Wno-unused %s

// sycl-no-diagnostics
void foo() {
  // notsycl-error@+1{{expected '(' for function-style cast or type construction}}
  __builtin_sycl_unique_stable_name(int);
}
