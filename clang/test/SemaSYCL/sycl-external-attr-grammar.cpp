// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// expected-error@+1{{'clang::sycl_external' attribute takes no arguments}}
[[clang::sycl_external(3)]] void bar() {}

// FIXME: this case should be diagnosed too
[[clang::sycl_external()]] void bad1();

// expected-error@+1{{expected expression}}
[[clang::sycl_external(,)]] void bad2();

// expected-error@+1{{'clang::sycl_external' attribute takes no arguments}}
[[clang::sycl_external(3)]] void bad3();

// expected-error@+1{{expected expression}}
[[clang::sycl_external(4,)]] void bad4();
