// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// expected-error@+1{{'sycl_external' attribute only applies to functions}}
[[clang::sycl_external]] int a;


// expected-error@+2{{'sycl_external' attribute only applies to functions}}
struct s {
[[clang::sycl_external]] int b;
};

// expected-error@+1{{'sycl_external' attribute takes no arguments}}
[[clang::sycl_external(3)]] void bar() {}

// FIXME: The first declaration of a function is required to have the attribute.
// The attribute may be optionally present on subsequent declarations
int foo(int c);

[[clang::sycl_external]] void foo();
