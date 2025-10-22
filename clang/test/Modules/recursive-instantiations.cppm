// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/type_traits.cppm -emit-module-interface -o %t/type_traits.pcm
// RUN: %clang_cc1 -std=c++20 %t/test.cpp -fprebuilt-module-path=%t -verify

//--- type_traits.cppm
export module type_traits;

export template <typename T>
constexpr bool is_pod_v = __is_pod(T);

//--- test.cpp
// expected-no-diagnostics
import type_traits;
// Base is either void or wrapper<T>.
template <class Base> struct wrapper : Base {};
template <> struct wrapper<void> {};

// wrap<0>::type<T> is wrapper<T>, wrap<1>::type<T> is wrapper<wrapper<T>>,
// and so on.
template <int N>
struct wrap {
  template <class Base>
  using type = wrapper<typename wrap<N-1>::template type<Base>>;
};

template <>
struct wrap<0> {
  template <class Base>
  using type = wrapper<Base>;
};

inline constexpr int kMaxRank = 40;
template <int N, class Base = void>
using rank = typename wrap<N>::template type<Base>;
using rank_selector_t = rank<kMaxRank>;

static_assert(is_pod_v<rank_selector_t>, "Must be POD");
