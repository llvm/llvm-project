// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// expected-warning@+1{{'sycl_external' attribute only applies to functions}}
[[clang::sycl_external]] int a;


// expected-warning@+2{{'sycl_external' attribute only applies to functions}}
struct s {
[[clang::sycl_external]] int b;
};

// FIXME: The first declaration of a function is required to have the attribute.
// The attribute may be optionally present on subsequent declarations
int foo(int c);

[[clang::sycl_external]] void foo();
