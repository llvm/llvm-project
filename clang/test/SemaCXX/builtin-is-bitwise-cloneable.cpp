// RUN: %clang_cc1 -fsyntax-only -verify %s
//
struct DynamicClass { virtual int Foo(); };
static_assert(!__is_trivially_copyable(DynamicClass));
static_assert(__is_bitwise_cloneable(DynamicClass));

struct InComplete; // expected-note{{forward declaration}}
static_assert(!__is_bitwise_cloneable(InComplete)); // expected-error{{incomplete type 'InComplete' used in type trait expression}}
