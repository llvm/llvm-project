// RUN: %check_clang_tidy -std=c++17-or-later %s readability-redundant-lambda-parentheses %t

int main() {
  // Basic cases - should warn
  auto a = []() { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto a = [] { return 42; };

  auto b = [x = 1]() { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto b = [x = 1] { return x; };

  auto c = []() {};
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto c = [] {};

  auto v = 1;
  auto call = [&v]() { return v; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto call = [&v] { return v; };

  // Should NOT warn - has parameters
  auto d = [](int x) { return x; };
  auto e = [](int x, int y) { return x + y; };

  // Should NOT warn under C++17/20 - has specifiers (only valid to remove in C++23+)
#if __cplusplus < 202302L
  auto f = []() mutable {};
  auto g = []() noexcept {};
  auto h = []() -> int { return 0; };
  auto i = []() constexpr { return 42; };
#endif

  // Should NOT warn - macro
#define LAMBDA []() { return 42; }
  auto k = LAMBDA;
#undef LAMBDA
}
