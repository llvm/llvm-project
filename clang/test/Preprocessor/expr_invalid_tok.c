// RUN: not %clang_cc1 -E %s 2>&1 | FileCheck %s
// PR2220

// CHECK: invalid token at start of a preprocessor expression
#if 1 * * 2
#endif

// CHECK: token is not a valid binary operator in a preprocessor subexpression
#if 4 [ 2
#endif


// PR2284 - The constant-expr production does not including comma.
// CHECK: [[@LINE+1]]:14: error: expected end of line in preprocessor expression
#if 1 ? 2 : 0, 1
#endif

// CHECK: [[@LINE+1]]:5: error: function-like macro 'FOO' is not defined
#if FOO(1, 2, 3)
#endif

// CHECK: [[@LINE+1]]:9: error: function-like macro 'BAR' is not defined
#if 1 + BAR(1, 2, 3)
#endif

// CHECK: [[@LINE+1]]:10: error: token is not a valid binary operator
#if (FOO)(1, 2, 3)
#endif
