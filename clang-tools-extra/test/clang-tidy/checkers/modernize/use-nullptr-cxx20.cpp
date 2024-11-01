// RUN: %check_clang_tidy -std=c++20 %s modernize-use-nullptr %t

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

class A {
  int a;
public:
  auto operator<=>(const A &other) const = default;
  // CHECK-FIXES: auto operator<=>(const A &other) const = default;
};

void test_cxx_rewritten_binary_ops() {
  A a1, a2;
  bool result;
  // should not change next line to (a1 nullptr a2)
  result = (a1 < a2);
  // CHECK-FIXES: result = (a1 < a2);
  // should not change next line to (a1 nullptr a2)
  result = (a1 >= a2);
  // CHECK-FIXES: result = (a1 >= a2);
  int *ptr = 0;
  // CHECK-FIXES: int *ptr = nullptr;
  result = (a1 > (ptr == 0 ? a1 : a2));
  // CHECK-FIXES: result = (a1 > (ptr == nullptr ? a1 : a2));
  result = (a1 > ((a1 > (ptr == 0 ? a1 : a2)) ? a1 : a2));
  // CHECK-FIXES: result = (a1 > ((a1 > (ptr == nullptr ? a1 : a2)) ? a1 : a2));
}

template<class T1, class T2>
struct P {
  T1 x1;
  T2 x2;
  friend auto operator<=>(const P&, const P&) = default;
  // CHECK-FIXES: friend auto operator<=>(const P&, const P&) = default;
};

bool foo(P<int,int> x, P<int, int> y) { return x < y; }
// CHECK-FIXES: bool foo(P<int,int> x, P<int, int> y) { return x < y; }
