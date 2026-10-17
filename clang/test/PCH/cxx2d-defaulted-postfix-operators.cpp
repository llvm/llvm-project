// RUN: %clang_cc1 -std=c++2d -verify -include %s %s
//
// RUN: %clang_cc1 -std=c++2d -emit-pch %s -o %t.pch
// RUN: %clang_cc1 -std=c++2d -include-pch %t.pch %s -verify
//
// RUN: %clang_cc1 -std=c++2d -emit-pch -fpch-instantiate-templates %s -o %t.pch
// RUN: %clang_cc1 -std=c++2d -include-pch %t.pch %s -verify

// expected-no-diagnostics

#ifndef INCLUDED
#define INCLUDED

struct Counter {
  int v;
  constexpr Counter &operator++() { ++v; return *this; }
  constexpr Counter &operator--() { --v; return *this; }
  Counter operator++(int) = default;
  friend Counter operator--(Counter &, int) = default;
};

// Ensure that we can round-trip DefaultedOrDeletedInfo through an AST file.
namespace LookupContext {
namespace A {
struct X { int v; };
}
namespace B {
constexpr A::X &operator++(A::X &x) { ++x.v; return x; }
}
namespace A {
using namespace B;
template <typename T> struct Y {
  T x;
  constexpr Y &operator++() { ++x; return *this; }
  Y operator++(int) = default;
};
} // namespace A
} // namespace LookupContext

#else

constexpr int counter() {
  Counter c{1};
  Counter a = c++;
  Counter b = c--;
  return a.v * 100 + b.v * 10 + c.v;
}
static_assert(counter() == 121);

namespace Other {
constexpr int lookup() {
  LookupContext::A::Y<LookupContext::A::X> y{{1}};
  LookupContext::A::Y<LookupContext::A::X> a = y++;
  return a.x.v * 10 + y.x.v;
}
static_assert(lookup() == 12);
} // namespace Other

#endif
