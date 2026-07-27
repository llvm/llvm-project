// RUN: %clang_cc1 -fsyntax-only -std=c++26 -verify %s

// expected-no-diagnostics

template<int (*a)[3]> void pr211930() {
  auto&& [x, y, z] = *a;
  // This shouldn't crash; x is value-dependent.
  constexpr int q = x;
}

struct S {int a = 4;};
namespace std {
    template <typename T> struct tuple_size;
    template <> struct tuple_size<S> { static const int value = 3; };
    template <> struct tuple_size<const S> { static const int value = 3; };
    template <int I, typename T> struct tuple_element;
    template <int I> struct tuple_element<I, S> {
        using type = const int;
    };
    template <int I> struct tuple_element<I, const S> {
        using type = const int;
    };
}
static const int Z = 4;
template<int x> constexpr const int &get(S&&s) { return s.a; }
template<int x> constexpr const int &get(const S&s) { return s.a; }
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
