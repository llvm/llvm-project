// RUN: %check_clang_tidy -std=c++20 %s modernize-use-nullptr %t

namespace std {
struct strong_ordering {
  int n;
  constexpr operator int() const { return n; }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
} // namespace std

class A {
public:
  auto operator<=>(const A &other) const = default;
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
