// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify=cxx20 %s
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
  // cxx20-note@-1 {{because 't.a.b' would be invalid: reference to non-static member function must be called}}

static_assert(C<Good>);
static_assert(!C<Bad>);
static_assert(C<Bad>); // cxx20-error {{static_assert failed}}
  // cxx20-note@-1 {{because 'DotFollowingFunctionName::Bad' does not satisfy 'C'}}
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
  // cxx20-note@-1 {{because 't.begin()' would be invalid: member reference type 'Holder<Incomplete> *' is a pointer}}

static_assert(C<Good>);
static_assert(!C<Bad>);
static_assert(C<Bad>); // cxx20-error {{static_assert failed}}
  // cxx20-note@-1 {{because 'DotFollowingPointer::Bad' (aka 'Holder<Incomplete> *') does not satisfy 'C'}}
#endif
} // namespace DotFollowingPointer
