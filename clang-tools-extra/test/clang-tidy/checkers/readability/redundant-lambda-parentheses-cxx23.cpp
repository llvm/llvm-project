// RUN: %check_clang_tidy -std=c++23 %s readability-redundant-lambda-parentheses %t

int main() {
  // Basic cases - should warn
  auto a = []() { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto a = [] { return 42; };{{$}}

  // Specifier cases - should also warn in C++23
  auto b = []() mutable {};
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto b = [] mutable {};{{$}}

  auto c = []() noexcept {};
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto c = [] noexcept {};{{$}}

  auto d = []() -> int { return 0; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto d = [] -> int { return 0; };{{$}}

  auto e = []() mutable noexcept {};
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto e = [] mutable noexcept {};{{$}}

  auto f = []() constexpr { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto f = [] constexpr { return 42; };{{$}}

  auto g = []() consteval { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES: {{^}}  auto g = [] consteval { return 42; };{{$}}

  // Should NOT warn - has parameters
  auto h = [](int x) { return x; };

  // Should NOT warn - macro
#define LAMBDA []() { return 42; }
  auto i = LAMBDA;
#undef LAMBDA
}
