// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

// libstdc++ 4.6.x contains a bug where it defines std::__atomic[0,1,2] as a
// non-inline namespace, then selects one of those namespaces and reopens it
// as inline, as a strange way of providing something like a using-directive.
// Clang has an egregious hack to work around the problem, by allowing a
// namespace to be converted from non-inline to inline in this one specific
// case.

// the last 4.6 release was 2013, so the hack is removed.  This checks __atomic
// is not special.
#ifdef BE_THE_HEADER

#pragma clang system_header

namespace std {
namespace __atomic0 { // expected-note {{previous definition}}
typedef int foobar;
} // namespace __atomic0
namespace __atomic1 {
typedef void foobar;
} // namespace __atomic1

inline namespace __atomic0 {} // expected-error {{cannot be reopened as inline}}
} // namespace std

#else

#define BE_THE_HEADER
#include "libstdcxx_atomic_ns_hack.cpp"

std::foobar fb; // expected-error {{no type named 'foobar' in namespace}}

#endif
