// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s readability-redundant-lambda-parentheses %t
// RUN: %check_clang_tidy -std=c++20 -check-suffixes=,CXX20 %s readability-redundant-lambda-parentheses %t
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffixes=,CXX20,CXX23 %s readability-redundant-lambda-parentheses %t


int main() {
  // Basic cases - warn in all standards
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

  // Should NOT warn - macro
#define LAMBDA []() { return 42; }
  auto k = LAMBDA;
#undef LAMBDA

  // Generic lambda - warns in C++20 and later
#if __cplusplus >= 202002L
  auto f = []<class T>() { return sizeof(T); };
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:23: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX20:   auto f = []<class T> { return sizeof(T); };

  // Should NOT warn - requires clause after parens
  auto g = []<class T>() requires (sizeof(T) > 0) {};

  // Should NOT warn - has parameters
  auto h = []<class T>(T x) { return x; };
#endif

  // Specifier cases - warn only in C++23, valid syntax in all standards
  auto i = []() mutable {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto i = [] mutable {};

  auto j = []() noexcept {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto j = [] noexcept {};

  auto l = []() -> int { return 0; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto l = [] -> int { return 0; };

  auto m = []() mutable noexcept {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto m = [] mutable noexcept {};

  auto n = []() constexpr { return 42; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto n = [] constexpr { return 42; };

  // consteval only valid in C++20+
#if __cplusplus >= 202002L
  auto o = []() consteval { return 42; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto o = [] consteval { return 42; };
#endif

  // Should NOT warn - attribute after parens
  auto p = []() [[]] {};

  // Should warn - attribute before parens
  auto q = [] [[]] () {};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto q = [] {{\[\[}}{{\]\]}} {};

  // Should NOT warn - attribute both before and after parens
  auto r = [] [[]] () [[]] {};

  auto s = []() noexcept [[]] {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX23:   auto s = [] noexcept {{\[\[}}{{\]\]}} {};

  auto t = []( /* comment */ ) { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto t = [] /* comment */ { return 42; };

  auto u = []() /* comment */ { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto u = [] /* comment */ { return 42; };

  auto w = [](/* comment */) { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto w = []/* comment */ { return 42; };

  auto x = [] /* comment */ () { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES:   auto x = [] /* comment */ { return 42; };

  // Should NOT warn - (void) is not truly empty parens
  auto void1 = [](void) {};
  auto void2 = [] (void) {};

  // Should NOT warn - macro between parens
#define EMPTY
#define VOID void
  auto macro1 = [](EMPTY) {};
  auto macro2 = [](VOID) {};
#undef EMPTY
#undef VOID

// Should NOT warn - macro for parens themselves
#define LPAREN (
#define RPAREN )
#define PARENS ()
  auto macro3 = []LPAREN RPAREN {};
  auto macro4 = []PARENS {};
#undef LPAREN
#undef RPAREN
#undef PARENS
}

#if __cplusplus >= 202002L
template <bool B>
void testRequires() {
  // Should NOT warn - requires clause after parens
  auto f1 = []() requires B {};
  auto f2 = []() noexcept requires B {};
  auto f3 = []<typename T>() requires B {};

  // Should warn - requires clause is before parens, parens are removable
  auto f4 = []<typename T> requires B () {};
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:39: warning: redundant empty parameter list in lambda expression [readability-redundant-lambda-parentheses]
  // CHECK-FIXES-CXX20:   auto f4 = []<typename T> requires B {};
}
#endif
