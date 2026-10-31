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

// Ensure that requires-clauses on defaulted postfix operators and the prefix
// operators they select round-trip through an AST file.
template <typename T> concept Small = sizeof(T) <= 4;
template <typename T> struct Constrained {
  T v;
  constexpr Constrained &operator++() requires Small<T> { v += 1; return *this; }
  constexpr Constrained &operator++() { v += 10; return *this; }
  constexpr Constrained &operator--() { --v; return *this; }
  Constrained operator++(int) requires Small<T> = default;
  friend Constrained operator--(Constrained &, int) requires Small<T> = default;
};
template <typename T> struct PrefixConstrained {
  T v;
  constexpr PrefixConstrained &operator++() requires Small<T> { ++v; return *this; }
  PrefixConstrained operator++(int) requires requires(PrefixConstrained &p) { ++p; } = default;
};

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

constexpr int constrained() {
  Constrained<int> c{1};
  Constrained<int> a = c++;
  Constrained<int> b = c--;
  return a.v * 100 + b.v * 10 + c.v;
}
static_assert(constrained() == 121);
template <typename T> concept PostIncrementable = requires(T t) { t++; };
static_assert(PostIncrementable<Constrained<int>>);
static_assert(!PostIncrementable<Constrained<long long>>);

constexpr int prefix_constrained() {
  PrefixConstrained<int> p{1};
  PrefixConstrained<int> a = p++;
  return a.v * 10 + p.v;
}
static_assert(prefix_constrained() == 12);
static_assert(!PostIncrementable<PrefixConstrained<long long>>);

#endif
