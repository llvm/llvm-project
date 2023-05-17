// RUN: %check_clang_tidy -std=c++17-or-later %s modernize-unary-static-assert %t

#define FOO static_assert(sizeof(a) <= 15, "");
#define MSG ""

void f_textless(int a) {
  static_assert(sizeof(a) <= 10, "");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use unary 'static_assert' when the string literal is an empty string [modernize-unary-static-assert]
  // CHECK-FIXES: {{^}}  static_assert(sizeof(a) <= 10 );{{$}}
  static_assert(sizeof(a) <= 12, L"");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use unary 'static_assert' when
  // CHECK-FIXES: {{^}}  static_assert(sizeof(a) <= 12 );{{$}}
  FOO
  // CHECK-FIXES: {{^}}  FOO{{$}}
  static_assert(sizeof(a) <= 17, MSG);
  // CHECK-FIXES: {{^}}  static_assert(sizeof(a) <= 17, MSG);{{$}}
}

void f_with_tex(int a) {
  static_assert(sizeof(a) <= 10, "Size of variable a is out of range!");
}

void f_unary(int a) { static_assert(sizeof(a) <= 10); }

void f_incorrect_assert() { static_assert(""); }
