// RUN: not %clang_cc1 -E %s 2>&1 | grep 'invalid token at start of a preprocessor expression'
// RUN: not %clang_cc1 -E %s 2>&1 | grep 'token is not a valid binary operator in a preprocessor subexpression'
// RUN: not %clang_cc1 -E %s 2>&1 | grep ':14: error: expected end of line in preprocessor expression'
// PR2220

#if 1 * * 2
#endif

#if 4 [ 2
#endif


// PR2284 - The constant-expr production does not including comma.
#if 1 ? 2 : 0, 1
#endif
