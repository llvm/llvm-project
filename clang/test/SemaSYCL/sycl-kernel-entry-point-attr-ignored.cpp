// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -verify %s

// These tests validate that the sycl_kernel_entry_point attribute is ignored
// when SYCL support is not enabled.

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

// expected-warning@+1 {{'sycl_kernel_entry_point' attribute ignored}}
[[clang::sycl_kernel_entry_point(KN<1>)]]
void ok1();

// expected-warning@+2 {{'sycl_kernel_entry_point' attribute ignored}}
template<typename KNT>
[[clang::sycl_kernel_entry_point(KNT)]]
void ok2() {}
template void ok2<KN<2>>();
