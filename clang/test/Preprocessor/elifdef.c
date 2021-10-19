// RUN: %clang_cc1 %s -Eonly -verify

#ifdef FOO
#elifdef BAR
#error "did not expect to get here"
#endif

/* expected-error@+4 {{"got it"}} */
#ifdef FOO
#elifdef BAR
#else
#error "got it"
#endif

/* expected-error@+3 {{"got it"}} */
#ifdef FOO
#elifndef BAR
#error "got it"
#endif

/* expected-error@+3 {{"got it"}} */
#ifdef FOO
#elifndef BAR
#error "got it"
#else
#error "did not expect to get here"
#endif

#define BAR
/* expected-error@+3 {{"got it"}} */
#ifdef FOO
#elifdef BAR
#error "got it"
#endif
#undef BAR

#ifdef FOO
#elifdef BAR // test that comments aren't an issue
#error "did not expect to get here"
#endif

/* expected-error@+4 {{"got it"}} */
#ifdef FOO
#elifdef BAR // test that comments aren't an issue
#else
#error "got it"
#endif

/* expected-error@+3 {{"got it"}} */
#ifdef FOO
#elifndef BAR // test that comments aren't an issue
#error "got it"
#endif

/* expected-error@+3 {{"got it"}} */
#ifdef FOO
#elifndef BAR // test that comments aren't an issue
#error "got it"
#else
#error "did not expect to get here"
#endif

#define BAR
/* expected-error@+3 {{"got it"}} */
#ifdef FOO
#elifdef BAR // test that comments aren't an issue
#error "got it"
#endif
#undef BAR

#define BAR
/* expected-error@+6 {{"got it"}} */
#ifdef FOO
#error "did not expect to get here"
#elifndef BAR
#error "did not expect to get here"
#else
#error "got it"
#endif
#undef BAR

/* expected-error@+3 {{#elifdef after #else}} */
#ifdef FOO
#else
#elifdef BAR
#endif

/* expected-error@+3 {{#elifndef after #else}} */
#ifdef FOO
#else
#elifndef BAR
#endif

#elifdef FOO /* expected-error {{#elifdef without #if}} */
#endif       /* expected-error {{#endif without #if}} */

#elifndef FOO /* expected-error {{#elifndef without #if}} */
#endif        /* expected-error {{#endif without #if}} */

/* Note, we do not expect errors about the missing macro name in the skipped
   blocks. This is consistent with #elif behavior. */
/* expected-error@+2 {{"got it"}} */
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
