// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

template <class>
struct Select;

template <template <class, class...> class C, class A, class... Rest>
struct Select<C<A, Rest...>> {
  static const int value = 1;
};

template <template <class> class C, class A>
struct Select<C<A>> {
  static const int value = 2;
};

template <class>
struct Unary;

template <class, class = float>
struct DefaultedBinary;

static_assert(Select<Unary<int>>::value == 2, "");
static_assert(Select<DefaultedBinary<int>>::value == 1, "");
