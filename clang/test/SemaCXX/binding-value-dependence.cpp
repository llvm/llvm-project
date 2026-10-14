// RUN: %clang_cc1 -fsyntax-only -std=c++26 -verify %s

template<int (*a)[3]> void pr211930() {
  auto&& [x, y, z] = *a;
  // This shouldn't crash; x is value-dependent.
  constexpr int q = x;

  // Variations using different forms of initialization.
  auto&& [x2, y2, z2]{*a};
  constexpr int q2 = x2;
  auto&& [x3, y3, z3](*a);
  constexpr int q3 = x3;
}

template<int (*a)[3]> void not_value_dependent() {
  auto [x, y, z] = *a;
  constexpr int c = &x == &x+1;
  switch (1) {
    case c:;  // expected-note {{previous case defined here}}
    case 0:;  // expected-error {{duplicate case value: 'c' and '0' both equal '0'}}
  }
}

struct S {int a = 4;};
struct S2 { };
namespace std {
    template <typename T> struct tuple_size;
    template <> struct tuple_size<S> { static const int value = 3; };
    template <> struct tuple_size<const S> { static const int value = 3; };
    template <> struct tuple_size<S2> { static const int value = 3; };
    template <int I, typename T> struct tuple_element;
    template <int I> struct tuple_element<I, S> {
        using type = const int;
    };
    template <int I> struct tuple_element<I, const S> {
        using type = const int;
    };
    template <int I> struct tuple_element<I, S2> {
        using type = const int;
    };
}
static const int Z = 4;
template<int x> constexpr const int &get(S&&s) { return s.a; }
template<int x> constexpr const int &get(const S&s) { return s.a; }
template<int x> constexpr const int get(const S2&s) { return 4; }
template<S *s> void value_dependent_get() {
  auto &[a,b,c] = *s;
  // This shouldn't warn: a is value-dependent.
  int rr[-11/(a)];
}
template<const S *s> void constexpr_value_dependent_get() {
  static constexpr auto [a,b,c] = *s;
  // This shouldn't warn: a is value-dependent.
  int rr[-11/(a)];
}
template<const S2 *s> void constexpr_non_value_dependent_get() {
  static auto [a,b,c] = *s;
  // The variable holding the initializer is not potentially-constant, so
  // the variable holding the return value of get() is a non-value-dependent
  // constant.
  switch (1) {
    case a: ; // expected-note {{previous case defined here}}
    case b: ; // expected-error {{duplicate case value: 'a' and 'b' both equal '4'}}
  }
}
