/* RUN: %clang_cc1 -E %s -DNO_ERRORS -Werror -Wundef
   RUN: not %clang_cc1 -E %s
 */

#ifdef NO_ERRORS
/* None of these divisions by zero are in live parts of the expression, do not
   emit any diagnostics. */

#define MACRO_0 0
#define MACRO_1 1

#if MACRO_0 && 10 / MACRO_0
foo
#endif

#if MACRO_1 || 10 / MACRO_0
bar
#endif

#if 0 ? 124/0 : 42
#endif

// PR2279
#if 0 ? 1/0: 2
#else
#error
#endif

// PR2279
#if 1 ? 2 ? 3 : 4 : 5
#endif

// PR2284
#if 1 ? 0: 1 ? 1/0: 1/0
#endif

#else


/* The 1/0 is live, it should error out. */
#if 0 && 1 ? 4 : 1 / 0
baz
#endif


#endif

// -Wundef should not warn about use of undefined identifier if not live.
#if (!defined(XXX) || XXX > 42)
#endif

