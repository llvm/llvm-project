// RUN: %clang_cc1 -fsyntax-only -verify %s
//
struct DynamicClass { virtual int Foo(); };
static_assert(!__is_trivially_copyable(DynamicClass));
static_assert(__is_bitwise_cloneable(DynamicClass));

struct InComplete; // expected-note{{forward declaration}}
static_assert(!__is_bitwise_cloneable(InComplete)); // expected-error{{incomplete type 'InComplete' used in type trait expression}}

// A struct with an incomplete field type is not bitwise cloneable.
struct Bar; // expected-note{{forward declaration}}
struct Foo {
  Bar bar; // expected-error{{field has incomplete type 'Bar'}}
};
static_assert(!__is_bitwise_cloneable(Foo));

// Don't crash when the type has a member of invalid/unknown type.
struct ABC { // expected-note {{'ABC' declared here}} expected-note {{definition of 'ABC' is not complete until the closing '}'}}
  ABCD ptr; // expected-error {{unknown type name 'ABCD'}} expected-error {{field has incomplete type 'ABC'}}
};
static_assert(!__is_bitwise_cloneable(ABC));
