/* RUN: %clang_cc1 -E %s 2>&1 >/dev/null | grep error: | count 3
 */

#ifdef

#endif

/* End of function-like macro invocation in #ifdef */
/* PR1936 */
#define f(x) x
#if f(2
#endif

int x;

