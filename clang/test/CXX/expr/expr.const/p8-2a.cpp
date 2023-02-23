// RUN: %clang_cc1 -std=c++20 -verify %s

// expected-no-diagnostics

namespace P1073R3 {
struct N {
  constexpr N() {}
  N(N const&) = delete;
};

template<typename T> constexpr void bad_assert_copyable() { T t; T t2 = t; }
using ineffective = decltype(bad_assert_copyable<N>());

// bad_assert_copyable<N> is not needed for constant evaluation
// (and thus not instantiated)
template<typename T> consteval void assert_copyable() { T t; T t2 = t; }
using check = decltype(assert_copyable<N>());
// FIXME: this should give an error because assert_copyable<N> is instantiated
// (because it is needed for constant evaluation), but the attempt to copy t is
// ill-formed.
} // namespace P1073R3

