// RUN: %clang_cc1 -frecovery-ast -verify %s
// RUN: %clang_cc1 -verify %s

struct alignas(invalid()) Foo {}; // expected-error {{use of undeclared identifier}}

constexpr int k = alignof(Foo);
