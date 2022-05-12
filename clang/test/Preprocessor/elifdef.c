// RUN: %clang_cc1 %s -Eonly -verify

/* expected-warning@+2 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#elifdef BAR
#error "did not expect to get here"
#endif

/* expected-error@+5 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#elifdef BAR
#else
#error "got it"
#endif

/* expected-error@+4 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifndef' directive is a C2x extension}} */
#ifdef FOO
#elifndef BAR
#error "got it"
#endif

/* expected-error@+4 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifndef' directive is a C2x extension}} */
#ifdef FOO
#elifndef BAR
#error "got it"
#else
#error "did not expect to get here"
#endif

#define BAR
/* expected-error@+4 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#elifdef BAR
#error "got it"
#endif
#undef BAR

/* expected-warning@+2 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#elifdef BAR // test that comments aren't an issue
#error "did not expect to get here"
#endif

/* expected-error@+5 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#elifdef BAR // test that comments aren't an issue
#else
#error "got it"
#endif

/* expected-error@+4 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifndef' directive is a C2x extension}} */
#ifdef FOO
#elifndef BAR // test that comments aren't an issue
#error "got it"
#endif

/* expected-error@+4 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifndef' directive is a C2x extension}} */
#ifdef FOO
#elifndef BAR // test that comments aren't an issue
#error "got it"
#else
#error "did not expect to get here"
#endif

#define BAR
/* expected-error@+4 {{"got it"}} */
/* expected-warning@+2 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#elifdef BAR // test that comments aren't an issue
#error "got it"
#endif
#undef BAR

#define BAR
/* expected-error@+7 {{"got it"}} */
/* expected-warning@+3 {{use of a '#elifndef' directive is a C2x extension}} */
#ifdef FOO
#error "did not expect to get here"
#elifndef BAR
#error "did not expect to get here"
#else
#error "got it"
#endif
#undef BAR

/* expected-error@+4 {{#elifdef after #else}} */
/* expected-warning@+3 {{use of a '#elifdef' directive is a C2x extension}} */
#ifdef FOO
#else
#elifdef BAR
#endif

/* expected-error@+4 {{#elifndef after #else}} */
/* expected-warning@+3 {{use of a '#elifndef' directive is a C2x extension}} */
#ifdef FOO
#else
#elifndef BAR
#endif

/* expected-warning@+1 {{use of a '#elifdef' directive is a C2x extension}} */
#elifdef FOO /* expected-error {{#elifdef without #if}} */
#endif       /* expected-error {{#endif without #if}} */

/* expected-warning@+1 {{use of a '#elifndef' directive is a C2x extension}} */
#elifndef FOO /* expected-error {{#elifndef without #if}} */
#endif        /* expected-error {{#endif without #if}} */

/* Note, we do not expect errors about the missing macro name in the skipped
   blocks. This is consistent with #elif behavior. */
/* expected-error@+4 {{"got it"}} */
/* expected-warning@+4 {{use of a '#elifdef' directive is a C2x extension}} */
/* expected-warning@+4 {{use of a '#elifndef' directive is a C2x extension}} */
#ifndef FOO
#error "got it"
#elifdef
#elifndef
#endif

/* expected-error@+3 {{#elif after #else}}*/
#if 1
#else
#elif
#endif
