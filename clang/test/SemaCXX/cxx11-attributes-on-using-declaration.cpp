// RUN: %clang_cc1 -pedantic -triple x86_64-apple-macos11 -std=c++20 -fsyntax-only -verify %s

static_assert(__has_extension(cxx_attributes_on_using_declarations), "");

namespace NS { typedef int x; }

[[clang::annotate("foo")]] using NS::x; // expected-warning{{ISO C++ does not allow an attribute list to appear here}}


[[deprecated]] using NS::x;                                    // expected-warning {{'deprecated' currently has no effect on a using declaration}} expected-warning{{ISO C++ does not allow}}
using NS::x [[deprecated]];                                    // expected-warning {{'deprecated' currently has no effect on a using declaration}} expected-warning{{ISO C++ does not allow}}
using NS::x __attribute__((deprecated));                       // expected-warning {{'deprecated' currently has no effect on a using declaration}}
using NS::x __attribute__((availability(macos,introduced=1))); // expected-warning {{'availability' currently has no effect on a using declaration}}

[[clang::availability(macos,introduced=1)]] using NS::x; // expected-warning {{'availability' currently has no effect on a using declaration}} expected-warning{{ISO C++ does not allow}}

// expected-warning@+1 3 {{ISO C++ does not allow an attribute list to appear here}}
[[clang::annotate("A")]] using NS::x [[clang::annotate("Y")]], NS::x [[clang::annotate("Z")]];

template <class T>
struct S : T {
  [[deprecated]] using typename T::x; // expected-warning{{ISO C++ does not allow}} expected-warning {{'deprecated' currently has no effect on a using declaration}}
  [[deprecated]] using T::y;          // expected-warning{{ISO C++ does not allow}} expected-warning {{'deprecated' currently has no effect on a using declaration}}

  using typename T::z [[deprecated]]; // expected-warning{{ISO C++ does not allow}} expected-warning {{'deprecated' currently has no effect on a using declaration}}
  using T::a [[deprecated]];          // expected-warning{{ISO C++ does not allow}} expected-warning {{'deprecated' currently has no effect on a using declaration}}
};

struct Base {};

template <class B>
struct DepBase1 : B {
  using B::B [[]];

};
template <class B>
struct DepBase2 : B {
  using B::B __attribute__(());
};

DepBase1<Base> db1;
DepBase2<Base> db2;
