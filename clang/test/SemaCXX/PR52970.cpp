// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics

struct Incomplete;
template <class T> struct Holder { T t; };

namespace DotFollowingFunctionName {
struct Good {
  struct Nested {
    int b;
  } a;
};

struct Bad {
  Holder<Incomplete> a();
};

template <class T>
constexpr auto f(T t) -> decltype((t.a.b, true)) { return true; }
constexpr bool f(...) { return false; }

static_assert(DotFollowingFunctionName::f(Good{}), "");
static_assert(!DotFollowingFunctionName::f(Bad{}), "");

#if __cplusplus >= 202002L
template <class T>
concept C = requires(T t) { t.a.b; };

static_assert(C<Good>);
static_assert(!C<Bad>);
#endif
} // namespace DotFollowingFunctionName

namespace DotFollowingPointer {
struct Good {
  int begin();
};
using Bad = Holder<Incomplete> *;

template <class T>
constexpr auto f(T t) -> decltype((t.begin(), true)) { return true; }
constexpr bool f(...) { return false; }

static_assert(DotFollowingPointer::f(Good{}), "");
static_assert(!DotFollowingPointer::f(Bad{}), "");

#if __cplusplus >= 202002L
template <class T>
concept C = requires(T t) { t.begin(); };

static_assert(C<Good>);
static_assert(!C<Bad>);
#endif
} // namespace DotFollowingPointer
