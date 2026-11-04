// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s readability-redundant-lambda-parameter-list %t
// RUN: %check_clang_tidy -std=c++20 -check-suffixes=,CXX20 %s readability-redundant-lambda-parameter-list %t
// RUN: %check_clang_tidy -std=c++23-or-later -check-suffixes=,CXX20,CXX23 %s readability-redundant-lambda-parameter-list %t


int main() {
  auto a = []() { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto a = [] { return 42; };

  auto b = [x = 1]() { return x; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto b = [x = 1] { return x; };

  auto c = []() {};
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto c = [] {};

  auto v = 1;
  auto call = [&v]() { return v; };
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto call = [&v] { return v; };

  auto d = [](int x) { return x; };
  auto e = [](int x, int y) { return x + y; };

#define LAMBDA []() { return 42; }
  auto k = LAMBDA;
#undef LAMBDA

#if __cplusplus >= 202002L
  auto f = []<class T>() { return sizeof(T); };
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:23: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX20:   auto f = []<class T> { return sizeof(T); };

  auto g = []<class T>() requires (sizeof(T) > 0) {};

  auto h = []<class T>(T x) { return x; };
#endif

  auto i = []() mutable {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto i = [] mutable {};

  auto j = []() noexcept {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto j = [] noexcept {};

  auto l = []() -> int { return 0; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto l = [] -> int { return 0; };

  auto m = []() mutable noexcept {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto m = [] mutable noexcept {};

  auto n = []() constexpr { return 42; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto n = [] constexpr { return 42; };

#if __cplusplus >= 202002L
  auto o = []() consteval { return 42; };
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto o = [] consteval { return 42; };
#endif

  auto p = []() [[]] {};

  auto q = [] [[]] () {};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto q = [] {{\[\[}}{{\]\]}} {};

  auto r = [] [[]] () [[]] {};

  auto s = []() noexcept [[]] {};
  // CHECK-MESSAGES-CXX23: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX23:   auto s = [] noexcept {{\[\[}}{{\]\]}} {};

  auto t = []( /* comment */ ) { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto t = [] /* comment */ { return 42; };

  auto u = []() /* comment */ { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto u = [] /* comment */ { return 42; };

  auto w = [](/* comment */) { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto w = []/* comment */ { return 42; };

  auto x = [] /* comment */ () { return 42; };
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES:   auto x = [] /* comment */ { return 42; };

  auto void1 = [](void) {};
  auto void2 = [] (void) {};

#define EMPTY
#define VOID void
  auto macro1 = [](EMPTY) {};
  auto macro2 = [](VOID) {};
#undef EMPTY
#undef VOID

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
  auto f1 = []() requires B {};
  auto f2 = []() noexcept requires B {};
  auto f3 = []<typename T>() requires B {};

  auto f4 = []<typename T> requires B () {};
  // CHECK-MESSAGES-CXX20: :[[@LINE-1]]:39: warning: redundant empty parameter list in lambda expression
  // CHECK-FIXES-CXX20:   auto f4 = []<typename T> requires B {};
}
#endif
