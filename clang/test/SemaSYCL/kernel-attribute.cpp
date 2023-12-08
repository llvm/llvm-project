// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl-is-device -verify %s

// Only function templates
[[clang::sycl_kernel]] int gv2 = 0; // expected-warning {{'sycl_kernel' attribute only applies to function templates}}
__attribute__((sycl_kernel)) int gv3 = 0; // expected-warning {{'sycl_kernel' attribute only applies to function templates}}

__attribute__((sycl_kernel)) void foo(); // expected-warning {{'sycl_kernel' attribute only applies to function templates}}
[[clang::sycl_kernel]] void foo1(); // expected-warning {{'sycl_kernel' attribute only applies to function templates}}

// Attribute takes no arguments
template <typename T, typename A>
__attribute__((sycl_kernel(1))) void foo(T P); // expected-error {{'sycl_kernel' attribute takes no arguments}}
template <typename T, typename A, int I>
[[clang::sycl_kernel(1)]] void foo1(T P);// expected-error {{'sycl_kernel' attribute takes no arguments}}

// At least two template parameters
template <typename T>
__attribute__((sycl_kernel)) void foo(T P); // expected-warning {{'sycl_kernel' attribute only applies to a function template with at least two template parameters}}
template <typename T>
[[clang::sycl_kernel]] void foo1(T P); // expected-warning {{'sycl_kernel' attribute only applies to a function template with at least two template parameters}}

// First two template parameters cannot be non-type template parameters
template <typename T, int A>
__attribute__((sycl_kernel)) void foo(T P); // expected-warning {{template parameter of a function template with the 'sycl_kernel' attribute cannot be a non-type template parameter}}
template <int A, typename T>
[[clang::sycl_kernel]] void foo1(T P); // expected-warning {{template parameter of a function template with the 'sycl_kernel' attribute cannot be a non-type template parameter}}

// Must return void
template <typename T, typename A>
__attribute__((sycl_kernel)) int foo(T P); // expected-warning {{function template with 'sycl_kernel' attribute must have a 'void' return type}}
template <typename T, typename A>
[[clang::sycl_kernel]] int foo1(T P); // expected-warning {{function template with 'sycl_kernel' attribute must have a 'void' return type}}

// Must take at least one argument
template <typename T, typename A>
__attribute__((sycl_kernel)) void foo(); // expected-warning {{function template with 'sycl_kernel' attribute must have a single parameter}}
template <typename T, typename A>
[[clang::sycl_kernel]] void foo1(T t, A a); // expected-warning {{function template with 'sycl_kernel' attribute must have a single parameter}}

// No diagnostics
template <typename T, typename A>
__attribute__((sycl_kernel)) void foo(T P);
template <typename T, typename A, int I>
[[clang::sycl_kernel]] void foo1(T P);
