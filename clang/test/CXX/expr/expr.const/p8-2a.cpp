// RUN: %clang_cc1 -std=c++20 -verify %s

// expected-no-diagnostics

namespace P1937R2 {
struct N {
  constexpr N() {}
  N(N const&) = delete;
};

template<typename T> constexpr void bad_assert_copyable() { T t; T t2 = t; }
using ineffective = decltype(bad_assert_copyable<N>());

template<typename T> consteval void assert_copyable() { T t; T t2 = t; }
// Prior to P1937R2 consteval functions were evaluated even in otherwise
// unevaluated context, now this is well-formed.
using check = decltype(assert_copyable<N>());

template<typename T>
__add_rvalue_reference(T) declval();

constexpr auto add1(auto lhs, auto rhs) {
    return lhs + rhs;
}
using T = decltype(add1(declval<int>(), declval<int>()));

consteval auto add2(auto lhs, auto rhs) {
    return lhs + rhs;
}
using T = decltype(add2(declval<int>(), declval<int>()));
} // namespace P1937R2

