// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify

// This is a test for an egregious hack in Clang that works around
// an issue with GCC's <type_traits> implementation. std::common_type
// relies on pre-standard rules for decltype(), in which it doesn't
// produce reference types so frequently.

#ifdef BE_THE_HEADER

#pragma GCC system_header
namespace std {
  template<typename T> T &&declval();

  template<typename...Ts> struct common_type {};
  template<typename A, typename B> struct common_type<A, B> {
    // Under the rules in the standard, this always produces a
    // reference type.
    typedef decltype(true ? declval<A>() : declval<B>()) type;
  };
}

#else

#define BE_THE_HEADER
#include "libstdcxx_common_type_hack.cpp"

using T = int;
using T = std::common_type<int, int>::type;

using U = int; // expected-note {{here}}
using U = decltype(true ? std::declval<int>() : std::declval<int>()); // expected-error {{different types}}

#endif
