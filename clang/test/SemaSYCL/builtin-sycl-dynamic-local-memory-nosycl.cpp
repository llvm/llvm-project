// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

void notSYCL() {
  // expected-error@+1 {{'__builtin_sycl_dynamic_local_memory' is only available in SYCL}}
  __builtin_sycl_dynamic_local_memory();
}
