// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

template <bool v>
struct BC {
  static constexpr bool value = v;
};

template <typename T, typename Arg>
struct Constructible : BC<__is_constructible(T, Arg)> {};

template <typename T>
using Requires = T::value;

template <typename T>
struct optional {
  template <typename U, Requires<Constructible<T, U>> = true>
  optional(U) {}
};

struct MO {};
struct S : MO {};
struct TB {
  TB(optional<S>) {}
};

class TD : TB, MO {
  using TB::TB;
};

void foo() {
  static_assert(Constructible<TD, TD>::value);
}
