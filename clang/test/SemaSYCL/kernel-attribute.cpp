// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -fcxx-exceptions -verify -pedantic -std=c++11 %s
// expected-no-diagnostics

__attribute((sycl_kernel)) void foo() {
}
