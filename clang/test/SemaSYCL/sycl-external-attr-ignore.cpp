// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-warning@+1{{'sycl_external' attribute ignored}}
[[clang::sycl_external]] void bar() {}

// expected-warning@+1{{'sycl_external' attribute ignored}}
[[clang::sycl_external]] int a;

// expected-warning@+2{{'sycl_external' attribute ignored}}
namespace not_sycl {
[[clang::sycl_external]] void foo() {}
}
