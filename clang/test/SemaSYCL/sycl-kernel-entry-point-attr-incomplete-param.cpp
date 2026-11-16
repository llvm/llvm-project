// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fsyntax-only -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -std=c++17 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -fsyntax-only -fsycl-is-host -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -std=c++20 -fsyntax-only -fsycl-is-device -verify %s

// Test that sycl_kernel_entry_point functions with parameters of incomplete
// type, or reference parameters that refer to incomplete types, are diagnosed
// rather than crashing the compiler.

template<int> struct KN;

template<typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts &&...) {}

struct S1; // expected-note 6{{forward declaration of 'S1'}}
struct S2 { int val; };

[[clang::sycl_kernel_entry_point(KN<1>)]]
void bad1(S1) {} // expected-error {{variable has incomplete type 'S1'}}

[[clang::sycl_kernel_entry_point(KN<2>)]]
void bad2(S1 &) {} // expected-error {{variable has incomplete type 'S1'}}

[[clang::sycl_kernel_entry_point(KN<3>)]]
void bad3(S1 &&) {} // expected-error {{variable has incomplete type 'S1'}}

[[clang::sycl_kernel_entry_point(KN<4>)]]
void bad4(const S1 &) {} // expected-error {{variable has incomplete type 'const S1'}}

[[clang::sycl_kernel_entry_point(KN<5>)]]
void bad5(S2, S1 &) {} // expected-error {{variable has incomplete type 'S1'}}

[[clang::sycl_kernel_entry_point(KN<6>)]]
void bad6(S1 &, int = 0) {} // expected-error {{variable has incomplete type 'S1'}}
