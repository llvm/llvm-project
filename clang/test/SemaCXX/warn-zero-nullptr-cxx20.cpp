// RUN: %clang_cc1 -fsyntax-only -verify %s -Wzero-as-null-pointer-constant -std=c++20

namespace std {
class strong_ordering;

// Mock how STD defined unspecified parameters for the operators below.
struct _CmpUnspecifiedParam {
  consteval
  _CmpUnspecifiedParam(int _CmpUnspecifiedParam::*) noexcept {}
};

struct strong_ordering {
  signed char value;

  friend constexpr bool operator==(strong_ordering v,
                                   _CmpUnspecifiedParam) noexcept {
    return v.value == 0;
  }
  friend constexpr bool operator<(strong_ordering v,
                                  _CmpUnspecifiedParam) noexcept {
    return v.value < 0;
  }
  friend constexpr bool operator>(strong_ordering v,
                                  _CmpUnspecifiedParam) noexcept {
    return v.value > 0;
  }
  friend constexpr bool operator>=(strong_ordering v,
                                   _CmpUnspecifiedParam) noexcept {
    return v.value >= 0;
  }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
} // namespace std

struct A {
  int a;
  constexpr auto operator<=>(const A &other) const = default;
};

void test_cxx_rewritten_binary_ops() {
  A a1, a2;
  bool result;
  result = (a1 < a2);
  result = (a1 >= a2);
  int *ptr = 0; // expected-warning{{zero as null pointer constant}}
  result = (a1 > (ptr == 0 ? a1 : a2)); // expected-warning{{zero as null pointer constant}}
  result = (a1 > ((a1 > (ptr == 0 ? a1 : a2)) ? a1 : a2)); // expected-warning{{zero as null pointer constant}}
}
