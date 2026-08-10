// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -pedantic -verify

#define UIE __attribute__((using_if_exists))

namespace NS {
typedef int x;
}

using NS::x __attribute__((using_if_exists));

using NS::x [[clang::using_if_exists]]; // expected-warning{{ISO C++ does not allow an attribute list to appear here}}

[[clang::using_if_exists]] // expected-warning{{ISO C++ does not allow an attribute list to appear here}}
using NS::not_there,
    NS::not_there2;

using NS::not_there3,                          // expected-error {{no member named 'not_there3' in namespace 'NS'}}
    NS::not_there4 [[clang::using_if_exists]]; // expected-warning{{C++ does not allow an attribute list to appear here}}

[[clang::using_if_exists]] using NS::not_there5 [[clang::using_if_exists]]; // expected-warning 2 {{ISO C++ does not allow}}

struct Base {};
struct Derived : Base {
  [[clang::using_if_exists]] using Base::x;          // expected-warning {{ISO C++ does not allow an attribute list to appear here}}
  using Base::y [[clang::using_if_exists]];          // expected-warning {{ISO C++ does not allow an attribute list to appear here}}
  [[clang::using_if_exists]] using Base::z, Base::q; // expected-warning {{C++ does not allow an attribute list to appear here}}
};
