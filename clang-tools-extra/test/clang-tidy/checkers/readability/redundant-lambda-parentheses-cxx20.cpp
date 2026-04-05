// RUN: %check_clang_tidy -std=c++20-or-later %s readability-redundant-lambda-parentheses %t

int main() {
  // Generic lambdas - should warn in C++20 and later
  auto a = []<class T>() { return sizeof(T); };
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto a = []<class T> { return sizeof(T); };

  auto b = []<class T>() requires true { return sizeof(T); };
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto b = []<class T> requires true { return sizeof(T); };

  // Should NOT warn - has parameters
  auto c = []<class T>(T x) { return x; };

  // Should NOT warn under C++20 - has specifiers (only valid to remove in C++23+)
#if __cplusplus < 202302L
  auto d = []<class T>() mutable { return sizeof(T); };
  auto e = []<class T>() noexcept { return sizeof(T); };
#endif
}
