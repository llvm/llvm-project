// RUN: %clang_cc1 -fsyntax-only -verify -std=c++26 %s

// expected-no-diagnostics

template <auto X, class = decltype(X)>
struct constant_wrapper;

template <class T>
concept constexpr_param = requires { typename constant_wrapper<T::value>; };

struct ops {
  template <constexpr_param T, constexpr_param R>
  constexpr auto operator+=(this T, R)
      -> constant_wrapper<(T::value += R::value)> {
    return {};
  }
};

template <auto X, class>
struct constant_wrapper : ops {
  static constexpr decltype(auto) value = (X);

  template <constexpr_param R>
  constexpr auto operator=(R) const -> constant_wrapper<(value = R::value)> {
    return {};
  }
};

struct A {
  int n;
  constexpr A(int n) : n(n) {}
  constexpr A operator=(A rhs) const { return A{rhs.n}; }
  constexpr A operator+=(A rhs) const { return A{n + rhs.n}; }
};

using X = constant_wrapper<A{1}>;
using Y = constant_wrapper<A{2}>;
using SimpleAssignment = decltype(X{} = Y{});
using CompoundAssignment = decltype(X{} += Y{});
