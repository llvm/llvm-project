// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-macro-to-enum %t -- -- -Wno-unused-value
// RUN: %clang -std=c++14 -Wno-unused-value -fsyntax-only %t.cpp

void use_two(int, int);

// The arguments are expressions, but the complete expansion is an argument
// list. Neither a partial argument nor the enclosing call validates it.
#define CALL_ARGS (1, 2)
// CHECK-FIXES: #define CALL_ARGS (1, 2)

void call_args() {
  use_two CALL_ARGS;
}

// Identical replacement tokens can instead supply a complete comma expression.
#define COMMA_EXPR (1, 2)
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'COMMA_EXPR' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: COMMA_EXPR = (1, 2)
// CHECK-FIXES-NEXT: };

int Comma = COMMA_EXPR;

void comma_uses() {
  use_two(COMMA_EXPR, COMMA_EXPR);
}
