// RUN: %clang_cc1 -std=c++26 -pedantic-errors -fsyntax-only -verify %s

template<class T, class U>
concept can_compare_eq = requires (const T& t, const U& u) { t == u; };

template<class T, class U>
concept can_compare_ne = requires (const T& t, const U& u) { t != u; };

template<class T, class U>
concept can_compare_lt = requires (const T& t, const U& u) { t < u; };

template<class T, class U>
concept can_compare_gt = requires (const T& t, const U& u) { t > u; };

template<class T, class U>
concept can_compare_le = requires (const T& t, const U& u) { t <= u; };

template<class T, class U>
concept can_compare_ge = requires (const T& t, const U& u) { t >= u; };

template<class T, class U>
concept can_compare_tw = requires (const T& t, const U& u) { t <=> u; };

static_assert(!can_compare_eq<int[1], int[1]>);
static_assert(!can_compare_ne<int[1], int[1]>);
static_assert(!can_compare_lt<int[1], int[1]>);
static_assert(!can_compare_gt<int[1], int[1]>);
static_assert(!can_compare_le<int[1], int[1]>);
static_assert(!can_compare_ge<int[1], int[1]>);
static_assert(!can_compare_tw<int[1], int[1]>);


void test() {
  int a[1]{}, b[1]{};
  [](const auto &x, const auto &y) {
    return x == y; // expected-error {{comparison between two arrays is ill-formed in C++26}}
  }(a, b); // expected-note {{in instantiation of function template specialization}}
}
