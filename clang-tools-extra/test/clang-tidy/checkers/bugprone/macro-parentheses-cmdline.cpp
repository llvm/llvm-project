// RUN: %check_clang_tidy %s bugprone-macro-parentheses %t -- -- -DVAL=0+0

// The previous command-line is producing warnings and fixes with the source
// locations from a virtual buffer. VAL is replaced by '0+0'.
// Fixes could not be applied and should not be reported.
int foo() { return VAL; }

#define V 0+0
int bar() { return V; }
// CHECK-FIXES: #define V (0+0)
