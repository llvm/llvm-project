// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// RUN: %clang_analyze_cc1 -DUSE_BUILTINS -analyzer-checker=core,unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s
// XFAIL: *

// This file is for tests that may eventually go into string.c, or may be
// deleted outright. At one point these tests passed, but only because we
// weren't correctly modelling the behavior of the relevant string functions.
// The tests aren't incorrect, but require the analyzer to be smarter about
// conjured values than it currently is.

//===----------------------------------------------------------------------===
// Declarations
//===----------------------------------------------------------------------===

// Some functions are so similar to each other that they follow the same code
// path, such as memcpy and __memcpy_chk, or memcmp and bcmp. If VARIANT is
// defined, make sure to use the variants instead to make sure they are still
// checked by the analyzer.

// Some functions are implemented as builtins. These should be #defined as
// BUILTIN(f), which will prepend "__builtin_" if USE_BUILTINS is defined.

// Functions that have variants and are also available as builtins should be
// declared carefully! See memcpy() for an example.

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else /* USE_BUILTINS */
# define BUILTIN(f) f
#endif /* USE_BUILTINS */

#define NULL 0
typedef typeof(sizeof(int)) size_t;

void clang_analyzer_eval(int);

//===----------------------------------------------------------------------===
// strnlen()
//===----------------------------------------------------------------------===

#define strnlen BUILTIN(strnlen)
size_t strnlen(const char *s, size_t maxlen);

void strnlen_liveness(const char *x) {
  if (strnlen(x, 10) < 5)
    return;
  clang_analyzer_eval(strnlen(x, 10) < 5); // expected-warning{{FALSE}}
}

void strnlen_subregion() {
  struct two_stringsn { char a[2], b[2]; };
  extern void use_two_stringsn(struct two_stringsn *);

  struct two_stringsn z;
  use_two_stringsn(&z);

  size_t a = strnlen(z.a, 10);
  z.b[0] = 5;
  size_t b = strnlen(z.a, 10);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  use_two_stringsn(&z);

  size_t c = strnlen(z.a, 10);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

extern void use_stringn(char *);
void strnlen_argument(char *x) {
  size_t a = strnlen(x, 10);
  size_t b = strnlen(x, 10);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  use_stringn(x);

  size_t c = strnlen(x, 10);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

extern char global_strn[];
void strnlen_global() {
  size_t a = strnlen(global_strn, 10);
  size_t b = strnlen(global_strn, 10);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  // Call a function with unknown effects, which should invalidate globals.
  use_stringn(0);

  size_t c = strnlen(global_strn, 10);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void strnlen_indirect(char *x) {
  size_t a = strnlen(x, 10);
  char *p = x;
  char **p2 = &p;
  size_t b = strnlen(x, 10);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  extern void use_stringn_ptr(char*const*);
  use_stringn_ptr(p2);

  size_t c = strnlen(x, 10);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}
