// RUN: %clang_cc1 -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++20 -include-pch %t -DTEST %s -verify
// RUN: %clang_cc1 -std=c++20 -include-pch %t -DTEST %s -verify -fexperimental-new-constant-interpreter

#ifndef TEST
namespace std {
enum class __order : signed char { less = -1, equal = 0, greater = 1 };

struct strong_ordering {
  __order value;

  constexpr explicit strong_ordering(__order value) : value(value) {}

  static const strong_ordering less;
  static const strong_ordering equal;
  static const strong_ordering greater;
};

inline constexpr strong_ordering strong_ordering::less(__order::less);
inline constexpr strong_ordering strong_ordering::equal(__order::equal);
inline constexpr strong_ordering strong_ordering::greater(__order::greater);
} // namespace std

template <class T, class U>
constexpr auto type_order = __builtin_type_order(T, U);
#else
static_assert(type_order<int, int>.value == std::__order::equal);
static_assert(type_order<int, long>.value != std::__order::equal);

// expected-no-diagnostics
#endif
