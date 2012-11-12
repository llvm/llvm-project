// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

// libstdc++ 4.6.x contains a bug where it defines std::__atomic[0,1,2] as a
// non-inline namespace, then selects one of those namespaces and reopens it
// as inline, as a strange way of providing something like a using-directive.
// Clang has an egregious hack to work around the problem, by allowing a
// namespace to be converted from non-inline to inline in this one specific
// case.

#ifdef BE_THE_HEADER

#pragma clang system_header

namespace std {
  namespace __atomic0 {
    typedef int foobar;
  }
  namespace __atomic1 {
    typedef void foobar;
  }

  inline namespace __atomic0 {}
}

#else

#define BE_THE_HEADER
#include "libstdcxx_atomic_ns_hack.cpp"

std::foobar fb;

using T = void; // expected-note {{here}}
using T = std::foobar; // expected-error {{different types ('std::foobar' (aka 'int') vs 'void')}}

#endif
