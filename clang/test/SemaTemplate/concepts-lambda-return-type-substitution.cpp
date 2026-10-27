// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++23 -verify %s
// RUN: %clang_cc1 -std=c++26 -verify %s
// expected-no-diagnostics

template <class T, class U>
concept DecaysTo = __is_same(__decay(T), U);

template <class T, auto Constraint>
concept Satisfies = requires { Constraint.template operator()<T>(); };

static_assert(requires(int i) {
  { i } -> Satisfies<[]<DecaysTo<int>> {}>;
});
static_assert(!requires(int i) {
  { i } -> Satisfies<[]<DecaysTo<double>> {}>;
});
