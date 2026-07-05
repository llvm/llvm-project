// RUN: %clang_cc1 -std=c++17 -verify %s

// expected-no-diagnostics

template <class T> struct add_restrict {
  typedef T __restrict type;
};

template <class T, class V> struct is_same {
  static constexpr bool value = false;
};

template <class T> struct is_same<T, T> {
  static constexpr bool value = true;
};

static_assert(is_same<int & __restrict, add_restrict<int &>::type>::value, "");
static_assert(is_same<int(), add_restrict<int()>::type>::value, "");
