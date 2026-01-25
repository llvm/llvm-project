// RUN: %check_clang_tidy %s bugprone-macro-parentheses %t -- -- -DVAL=0+0

int foo() { return VAL; }
// CHECK-MESSAGES: warning: macro replacement list should be enclosed in parentheses; macro 'VAL' defined as '0+0' [bugprone-macro-parentheses]

#define V 0+0
int bar() { return V; }
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: macro replacement list should be enclosed in parentheses [bugprone-macro-parentheses]
// CHECK-FIXES: #define V (0+0)
