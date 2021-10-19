/* RUN: %clang_cc1 -E -verify %s
 */

/* expected-error@+1 {{macro name missing}} */
#ifdef
#endif

/* expected-error@+1 {{macro name must be an identifier}} */
#ifdef !
#endif

/* expected-error@+1 {{macro name missing}} */
#if defined
#endif

/* PR1936 */
/* expected-error@+2 {{unterminated function-like macro invocation}} expected-error@+2 {{expected value in expression}} expected-note@+1 {{macro 'f' defined here}} */
#define f(x) x
#if f(2
#endif

/* expected-error@+2 {{macro name missing}} */
#ifdef FOO
#elifdef
#endif

/* expected-error@+2 {{macro name must be an identifier}} */
#ifdef FOO
#elifdef !
#endif

int x;
