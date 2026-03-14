/* RUN: %clang_cc1 %s -Eonly -std=c89 -pedantic -verify
*/
// RUN: %clang_cc1 %s -Eonly -std=c89 -pedantic -Wno-gnu-zero-variadic-macro-arguments -verify -DOMIT_VARIADIC_MACRO_ARGS -DVARIADIC_MACRO_ARGS_REMOVE_COMMA
// RUN: %clang_cc1 %s -Eonly -std=c89 -pedantic -Wno-variadic-macro-arguments-omitted -verify -DOMIT_VARIADIC_MACRO_ARGS
/* PR3937 */
#define zero() 0 /* expected-note 2 {{defined here}} */
#define one(x) 0 /* expected-note 2 {{defined here}} */
#define two(x, y) 0 /* expected-note 4 {{defined here}} */
#define zero_dot(...) 0   /* expected-warning {{variadic macros are a C99 feature}} */
#define one_dot(x, ...) 0 /* expected-warning {{variadic macros are a C99 feature}} */

#ifndef OMIT_VARIADIC_MACRO_ARGS
/* expected-note@-3 2{{macro 'one_dot' defined here}} */
#endif

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



/* PR4006 */
#define e(...) __VA_ARGS__  /* expected-warning {{variadic macros are a C99 feature}} */
e(x)
e()

zero_dot()
one_dot(x)  /* empty ... argument */
one_dot()   /* empty first argument, elided ... */

#ifndef OMIT_VARIADIC_MACRO_ARGS
/* expected-warning@-4 {{passing no argument for the '...' parameter of a variadic macro is a C23 extension}} */
/* expected-warning@-4 {{passing no argument for the '...' parameter of a variadic macro is a C23 extension}} */
#endif

/* Crash with function-like macro test at end of directive. */
#define E() (i == 0)
#if E
#endif

#define NSAssert(condition, desc, ...) /* expected-warning {{variadic macros are a C99 feature}} */ \
    SomeComplicatedStuff((desc), ##__VA_ARGS__)

#ifndef VARIADIC_MACRO_ARGS_REMOVE_COMMA
/* expected-warning@-3 {{token pasting of ',' and __VA_ARGS__ is a GNU extension}} */
#endif

NSAssert(somecond, somedesc)
