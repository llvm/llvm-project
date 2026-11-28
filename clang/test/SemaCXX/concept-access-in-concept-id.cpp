// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s

template <class T, class U> concept same_as = __is_same(T, U);
template <class T> T &&declval();

template <class T>
concept has_int_x = same_as<decltype(declval<T>().x), int>;

template <class T>
using x_t = decltype(declval<T>().x);

template <class T>
concept has_int_x_via_alias = same_as<x_t<T>, int>;

class A {
  int x;
};
class B {
public:
  int x;
};

static_assert(!has_int_x<A>);
static_assert(has_int_x<B>);
static_assert(!has_int_x_via_alias<A>);
static_assert(has_int_x_via_alias<B>);
