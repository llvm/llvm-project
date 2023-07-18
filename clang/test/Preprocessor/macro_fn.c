/* RUN: %clang_cc1 %s -Eonly -std=c89 -pedantic -verify
*/
/* PR3937 */
#define zero() 0 /* expected-note 2 {{defined here}} */
#define one(x) 0 /* expected-note 2 {{defined here}} */
#define two(x, y) 0 /* expected-note 4 {{defined here}} */
#define zero_dot(...) 0   /* expected-warning {{variadic macros are a C99 feature}} */
#define one_dot(x, ...) 0 /* expected-warning {{variadic macros are a C99 feature}} expected-note 2{{macro 'one_dot' defined here}} */

zero()
zero(1);          /* expected-error {{too many arguments provided to function-like macro invocation}} */
zero(1, 2, 3);    /* expected-error {{too many arguments provided to function-like macro invocation}} */

one()   /* ok */
one(a)
one(a,)           /* expected-error {{too many arguments provided to function-like macro invocation}} \
                     expected-warning {{empty macro arguments are a C99 feature}}*/
one(a, b)         /* expected-error {{too many arguments provided to function-like macro invocation}} */

two()       /* expected-error {{too few arguments provided to function-like macro invocation}} */
two(a)      /* expected-error {{too few arguments provided to function-like macro invocation}} */
two(a,b)
two(a, )    /* expected-warning {{empty macro arguments are a C99 feature}} */
two(a,b,c)  /* expected-error {{too many arguments provided to function-like macro invocation}} */
two(
    ,     /* expected-warning {{empty macro arguments are a C99 feature}} */
    ,     /* expected-warning {{empty macro arguments are a C99 feature}}  \
             expected-error {{too many arguments provided to function-like macro invocation}} */
    )     /* expected-warning {{empty macro arguments are a C99 feature}} */
two(,)      /* expected-warning 2 {{empty macro arguments are a C99 feature}} */



/* PR4006 & rdar://6807000 */
#define e(...) __VA_ARGS__  /* expected-warning {{variadic macros are a C99 feature}} */
e(x)
e()

zero_dot()
one_dot(x)  /* empty ... argument: expected-warning {{must specify at least one argument for '...' parameter of variadic macro}}  */
one_dot()   /* empty first argument, elided ...: expected-warning {{must specify at least one argument for '...' parameter of variadic macro}} */


/* rdar://6816766 - Crash with function-like macro test at end of directive. */
#define E() (i == 0)
#if E
#endif


/* <rdar://problem/12292192> */
#define NSAssert(condition, desc, ...) /* expected-warning {{variadic macros are a C99 feature}} */ \
    SomeComplicatedStuff((desc), ##__VA_ARGS__) /* expected-warning {{token pasting of ',' and __VA_ARGS__ is a GNU extension}} */
NSAssert(somecond, somedesc)
