// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-trailing-return-type %t -- -- -fno-delayed-template-parsing

namespace std {
template <typename T, typename U>
struct is_same { static constexpr auto value = false; };

template <typename T>
struct is_same<T, T> { static constexpr auto value = true; };

template <typename T>
concept floating_point = std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, long double>::value;
}

void test_lambda_positive() {
  auto l1 = []<typename T, typename U>(T x, U y) { return x + y; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l1 = []<typename T, typename U>(T x, U y) -> auto { return x + y; };
  auto l2 = [](auto x) requires requires { x + x; } { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l2 = [](auto x) -> auto requires requires { x + x; } { return x; };
  auto l3 = [](auto x) requires std::floating_point<decltype(x)> { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l3 = [](auto x) -> auto requires std::floating_point<decltype(x)> { return x; };
  auto l4 = [](int x) consteval { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l4 = [](int x) consteval -> int { return x; };
  // Complete complex example
  auto l5 = []<typename T, typename U>(T x, U y) constexpr noexcept
    requires std::floating_point<T> && std::floating_point<U>
  { return x * y; };
  // CHECK-MESSAGES: :[[@LINE-3]]:13: warning: use a trailing return type for this lambda [modernize-use-trailing-return-type]
  // CHECK-FIXES: auto l5 = []<typename T, typename U>(T x, U y) constexpr noexcept
  // CHECK-FIXES:   -> auto requires std::floating_point<T> && std::floating_point<U>
  // CHECK-FIXES: { return x * y; };
}
