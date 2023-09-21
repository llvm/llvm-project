// RUN: %clang_analyze_cc1 -verify %s -Wno-null-dereference \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// RUN: %clang_analyze_cc1 -verify %s -Wno-null-dereference -DUSE_BUILTINS \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// RUN: %clang_analyze_cc1 -verify %s -Wno-null-dereference -DVARIANT \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// RUN: %clang_analyze_cc1 -verify %s -Wno-null-dereference \
// RUN:   -DUSE_BUILTINS -DVARIANT \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.security.taint \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// RUN: %clang_analyze_cc1 -verify %s -Wno-null-dereference \
// RUN:   -DSUPPRESS_OUT_OF_BOUND \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap \
// RUN:   -analyzer-checker=alpha.unix.cstring.NotNullTerminated \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

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

int scanf(const char *restrict format, ...);
void *malloc(size_t);
void free(void *);
void *memcpy(void *dest, const void *src, size_t n);

//===----------------------------------------------------------------------===
// strlen()
//===----------------------------------------------------------------------===

#define strlen BUILTIN(strlen)
size_t strlen(const char *s);

void strlen_constant0(void) {
  clang_analyzer_eval(strlen("123") == 3); // expected-warning{{TRUE}}
}

void strlen_constant1(void) {
  const char *a = "123";
  clang_analyzer_eval(strlen(a) == 3); // expected-warning{{TRUE}}
}

void strlen_constant2(char x) {
  char a[] = "123";
  clang_analyzer_eval(strlen(a) == 3); // expected-warning{{TRUE}}

  a[0] = x;
  clang_analyzer_eval(strlen(a) == 3); // expected-warning{{UNKNOWN}}
}

size_t strlen_null(void) {
  return strlen(0); // expected-warning{{Null pointer passed as 1st argument to string length function}}
}

size_t strlen_fn(void) {
  return strlen((char*)&strlen_fn); // expected-warning{{Argument to string length function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

size_t strlen_nonloc(void) {
label:
  return strlen((char*)&&label); // expected-warning{{Argument to string length function is the address of the label 'label', which is not a null-terminated string}}
}

void strlen_subregion(void) {
  struct two_strings { char a[2], b[2]; };
  extern void use_two_strings(struct two_strings *);

  struct two_strings z;
  use_two_strings(&z);

  size_t a = strlen(z.a);
  z.b[0] = 5;
  size_t b = strlen(z.a);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  use_two_strings(&z);

  size_t c = strlen(z.a);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

extern void use_string(char *);
void strlen_argument(char *x) {
  size_t a = strlen(x);
  size_t b = strlen(x);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  use_string(x);

  size_t c = strlen(x);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

extern char global_str[];
void strlen_global(void) {
  size_t a = strlen(global_str);
  size_t b = strlen(global_str);
  if (a == 0) {
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
    // Make sure clang_analyzer_eval does not invalidate globals.
    clang_analyzer_eval(strlen(global_str) == 0); // expected-warning{{TRUE}}
  }

  // Call a function with unknown effects, which should invalidate globals.
  use_string(0);

  size_t c = strlen(global_str);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void strlen_indirect(char *x) {
  size_t a = strlen(x);
  char *p = x;
  char **p2 = &p;
  size_t b = strlen(x);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  extern void use_string_ptr(char*const*);
  use_string_ptr(p2);

  size_t c = strlen(x);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void strlen_indirect2(char *x) {
  size_t a = strlen(x);
  char *p = x;
  char **p2 = &p;
  extern void use_string_ptr2(char**);
  use_string_ptr2(p2);

  size_t c = strlen(x);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void strlen_liveness(const char *x) {
  if (strlen(x) < 5)
    return;
  clang_analyzer_eval(strlen(x) < 5); // expected-warning{{FALSE}}
}


size_t strlenWrapper(const char *str) {
  return strlen(str);
}

extern void invalidate(char *s);

void testStrlenCallee(void) {
  char str[42];
  invalidate(str);
  size_t lenBefore = strlenWrapper(str);
  invalidate(str);
  size_t lenAfter = strlenWrapper(str);
  clang_analyzer_eval(lenBefore == lenAfter); // expected-warning{{UNKNOWN}}
}


//===----------------------------------------------------------------------===
// strnlen()
//===----------------------------------------------------------------------===

size_t strnlen(const char *s, size_t maxlen);

void strnlen_constant0(void) {
  clang_analyzer_eval(strnlen("123", 10) == 3); // expected-warning{{TRUE}}
}

void strnlen_constant1(void) {
  const char *a = "123";
  clang_analyzer_eval(strnlen(a, 10) == 3); // expected-warning{{TRUE}}
}

void strnlen_constant2(char x) {
  char a[] = "123";
  clang_analyzer_eval(strnlen(a, 10) == 3); // expected-warning{{TRUE}}
  a[0] = x;
  clang_analyzer_eval(strnlen(a, 10) == 3); // expected-warning{{UNKNOWN}}
}

void strnlen_constant4(void) {
  clang_analyzer_eval(strnlen("123456", 3) == 3); // expected-warning{{TRUE}}
}

void strnlen_constant5(void) {
  const char *a = "123456";
  clang_analyzer_eval(strnlen(a, 3) == 3); // expected-warning{{TRUE}}
}

void strnlen_constant6(char x) {
  char a[] = "123456";
  clang_analyzer_eval(strnlen(a, 3) == 3); // expected-warning{{TRUE}}
  a[0] = x;
  clang_analyzer_eval(strnlen(a, 3) == 3); // expected-warning{{UNKNOWN}}
}

size_t strnlen_null(void) {
  return strnlen(0, 3); // expected-warning{{Null pointer passed as 1st argument to string length function}}
}

size_t strnlen_fn(void) {
  return strnlen((char*)&strlen_fn, 3); // expected-warning{{Argument to string length function is the address of the function 'strlen_fn', which is not a null-terminated string}}
}

size_t strnlen_nonloc(void) {
label:
  return strnlen((char*)&&label, 3); // expected-warning{{Argument to string length function is the address of the label 'label', which is not a null-terminated string}}
}

void strnlen_zero(void) {
  clang_analyzer_eval(strnlen("abc", 0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strnlen(NULL, 0) == 0); // expected-warning{{TRUE}}
}

size_t strnlen_compound_literal(void) {
  // This used to crash because we don't model the string lengths of
  // compound literals.
  return strnlen((char[]) { 'a', 'b', 0 }, 1);
}

size_t strnlen_unknown_limit(float f) {
  // This used to crash because we don't model the integer values of floats.
  return strnlen("abc", (int)f);
}

void strnlen_is_not_strlen(char *x) {
  clang_analyzer_eval(strnlen(x, 10) == strlen(x)); // expected-warning{{UNKNOWN}}
}

void strnlen_at_limit(char *x) {
  size_t len = strnlen(x, 10);
  clang_analyzer_eval(len <= 10); // expected-warning{{TRUE}}
  clang_analyzer_eval(len == 10); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(len < 10); // expected-warning{{UNKNOWN}}
}

void strnlen_at_actual(size_t limit) {
  size_t len = strnlen("abc", limit);
  clang_analyzer_eval(len <= 3); // expected-warning{{TRUE}}
  // This is due to eager assertion in strnlen.
  if (limit == 0) {
    clang_analyzer_eval(len == 0); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(len == 3); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(len < 3); // expected-warning{{UNKNOWN}}
  }
}

//===----------------------------------------------------------------------===
// strcpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strcpy_chk BUILTIN(__strcpy_chk)
char *__strcpy_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define strcpy(a,b) __strcpy_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define strcpy BUILTIN(strcpy)
char *strcpy(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */


void strcpy_null_dst(char *x) {
  strcpy(NULL, x); // expected-warning{{Null pointer passed as 1st argument to string copy function}}
}

void strcpy_null_src(char *x) {
  strcpy(x, NULL); // expected-warning{{Null pointer passed as 2nd argument to string copy function}}
}

void strcpy_fn(char *x) {
  strcpy(x, (char*)&strcpy_fn); // expected-warning{{Argument to string copy function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
}

void strcpy_fn_const(char *x) {
  strcpy(x, (const char*)&strcpy_fn); // expected-warning{{Argument to string copy function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
}

extern int globalInt;
void strcpy_effects(char *x, char *y) {
  char a = x[0];
  if (globalInt != 42)
    return;

  clang_analyzer_eval(strcpy(x, y) == x); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(x) == strlen(y)); // expected-warning{{TRUE}}
  clang_analyzer_eval(a == x[0]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
void strcpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strcpy(x, y); // expected-warning{{String copy function overflows the destination buffer}}
}
#endif

void strcpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strcpy(x, y); // no-warning
}

// PR37503
void *get_void_ptr(void);
char ***type_punned_ptr;
void strcpy_no_assertion(char c) {
  *(unsigned char **)type_punned_ptr = (unsigned char *)(get_void_ptr());
  strcpy(**type_punned_ptr, &c); // no-crash
}

// PR49007
char f(char ***c, int *i) {
  *(void **)c = i + 1;
  return (**c)[0]; // no-crash
}

//===----------------------------------------------------------------------===
// stpcpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __stpcpy_chk BUILTIN(__stpcpy_chk)
char *__stpcpy_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define stpcpy(a,b) __stpcpy_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define stpcpy BUILTIN(stpcpy)
char *stpcpy(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */


void stpcpy_effect(char *x, char *y) {
  char a = x[0];

  clang_analyzer_eval(stpcpy(x, y) == &x[strlen(y)]); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(x) == strlen(y)); // expected-warning{{TRUE}}
  clang_analyzer_eval(a == x[0]); // expected-warning{{UNKNOWN}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
void stpcpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    stpcpy(x, y); // expected-warning{{String copy function overflows the destination buffer}}
}
#endif

void stpcpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    stpcpy(x, y); // no-warning
}

//===----------------------------------------------------------------------===
// strcat()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strcat_chk BUILTIN(__strcat_chk)
char *__strcat_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define strcat(a,b) __strcat_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define strcat BUILTIN(strcat)
char *strcat(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */


void strcat_null_dst(char *x) {
  strcat(NULL, x); // expected-warning{{Null pointer passed as 1st argument to string concatenation function}}
}

void strcat_null_src(char *x) {
  strcat(x, NULL); // expected-warning{{Null pointer passed as 2nd argument to string concatenation function}}
}

void strcat_fn(char *x) {
  strcat(x, (char*)&strcat_fn); // expected-warning{{Argument to string concatenation function is the address of the function 'strcat_fn', which is not a null-terminated string}}
}

void strcat_effects(char *y) {
  char x[8] = "123";
  size_t orig_len = strlen(x);
  char a = x[0];

  if (strlen(y) != 4)
    return;

  clang_analyzer_eval(strcat(x, y) == x); // expected-warning{{TRUE}}
  clang_analyzer_eval((int)strlen(x) == (orig_len + strlen(y))); // expected-warning{{TRUE}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
void strcat_overflow_0(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strcat(x, y); // expected-warning{{String concatenation function overflows the destination buffer}}
}

void strcat_overflow_1(char *y) {
  char x[4] = "12";
  if (strlen(y) == 3)
    strcat(x, y); // expected-warning{{String concatenation function overflows the destination buffer}}
}

void strcat_overflow_2(char *y) {
  char x[4] = "12";
  if (strlen(y) == 2)
    strcat(x, y); // expected-warning{{String concatenation function overflows the destination buffer}}
}
#endif

void strcat_no_overflow(char *y) {
  char x[5] = "12";
  if (strlen(y) == 2)
    strcat(x, y); // no-warning
}

void strcat_symbolic_dst_length(char *dst) {
	strcat(dst, "1234");
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}
}

void strcat_symbolic_dst_length_taint(char *dst) {
  scanf("%s", dst); // Taint data.
  strcat(dst, "1234");
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}
}

void strcat_unknown_src_length(char *src, int offset) {
	char dst[8] = "1234";
	strcat(dst, &src[offset]);
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}
}

// There is no strcat_unknown_dst_length because if we can't get a symbolic
// length for the "before" strlen, we won't be able to set one for "after".

void strcat_too_big(char *dst, char *src) {
  // We assume this can never actually happen, so we don't get a warning.
	if (strlen(dst) != (((size_t)0) - 2))
		return;
	if (strlen(src) != 2)
		return;
	strcat(dst, src);
}


//===----------------------------------------------------------------------===
// strncpy()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strncpy_chk BUILTIN(__strncpy_chk)
char *__strncpy_chk(char *restrict s1, const char *restrict s2, size_t n, size_t destlen);

#define strncpy(a,b,n) __strncpy_chk(a,b,n,(size_t)-1)

#else /* VARIANT */

#define strncpy BUILTIN(strncpy)
char *strncpy(char *restrict s1, const char *restrict s2, size_t n);

#endif /* VARIANT */


void strncpy_null_dst(char *x) {
  strncpy(NULL, x, 5); // expected-warning{{Null pointer passed as 1st argument to string copy function}}
}

void strncpy_null_src(char *x) {
  strncpy(x, NULL, 5); // expected-warning{{Null pointer passed as 2nd argument to string copy function}}
}

void strncpy_fn(char *x) {
  strncpy(x, (char*)&strcpy_fn, 5); // expected-warning{{Argument to string copy function is the address of the function 'strcpy_fn', which is not a null-terminated string}}
}

void strncpy_effects(char *x, char *y) {
  char a = x[0];

  clang_analyzer_eval(strncpy(x, y, 5) == x); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(x) == strlen(y)); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(a == x[0]); // expected-warning{{UNKNOWN}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
// Enabling the malloc checker enables some of the buffer-checking portions
// of the C-string checker.
void cstringchecker_bounds_nocrash(void) {
  char *p = malloc(2);
  strncpy(p, "AAA", sizeof("AAA"));
  // expected-warning@-1 {{String copy function overflows the destination buffer}}
  free(p);
}

void strncpy_overflow(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strncpy(x, y, 5);
    // expected-warning@-1 {{String copy function overflows the destination buffer}}
#ifndef VARIANT
    // expected-warning@-3 {{size argument is too large; destination buffer has size 4, but size argument is 5}}
#endif
}

void strncpy_no_overflow(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, 5);
    // expected-warning@-1 {{String copy function overflows the destination buffer}}
#ifndef VARIANT
    // expected-warning@-3 {{size argument is too large; destination buffer has size 4, but size argument is 5}}
#endif
}

void strncpy_no_overflow2(char *y, int n) {
	if (n <= 4)
		return;

  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, n);
  // expected-warning@-1 {{String copy function overflows the destination buffer}}
}
#endif

void strncpy_truncate(char *y) {
  char x[4];
  if (strlen(y) == 4)
    strncpy(x, y, 3); // no-warning
}

void strncpy_no_truncate(char *y) {
  char x[4];
  if (strlen(y) == 3)
    strncpy(x, y, 3); // no-warning
}

void strncpy_exactly_matching_buffer(char *y) {
	char x[4];
	strncpy(x, y, 4); // no-warning

	// strncpy does not null-terminate, so we have no idea what the strlen is
	// after this.
  clang_analyzer_eval(strlen(x) > 4); // expected-warning{{UNKNOWN}}
}

void strncpy_zero(char *src) {
  char dst[] = "123";
  strncpy(dst, src, 0); // no-warning
}

void strncpy_empty(void) {
  char dst[] = "123";
  char src[] = "";
  strncpy(dst, src, 4); // no-warning
}

//===----------------------------------------------------------------------===
// strncat()
//===----------------------------------------------------------------------===

#ifdef VARIANT

#define __strncat_chk BUILTIN(__strncat_chk)
char *__strncat_chk(char *restrict s1, const char *restrict s2, size_t n, size_t destlen);

#define strncat(a,b,c) __strncat_chk(a,b,c, (size_t)-1)

#else /* VARIANT */

#define strncat BUILTIN(strncat)
char *strncat(char *restrict s1, const char *restrict s2, size_t n);

#endif /* VARIANT */


void strncat_null_dst(char *x) {
  strncat(NULL, x, 4); // expected-warning{{Null pointer passed as 1st argument to string concatenation function}}
}

void strncat_null_src(char *x) {
  strncat(x, NULL, 4); // expected-warning{{Null pointer passed as 2nd argument to string concatenation function}}
}

void strncat_fn(char *x) {
  strncat(x, (char*)&strncat_fn, 4); // expected-warning{{Argument to string concatenation function is the address of the function 'strncat_fn', which is not a null-terminated string}}
}

void strncat_effects(char *y) {
  char x[8] = "123";
  size_t orig_len = strlen(x);
  char a = x[0];

  if (strlen(y) != 4)
    return;

  clang_analyzer_eval(strncat(x, y, strlen(y)) == x); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(x) == (orig_len + strlen(y))); // expected-warning{{TRUE}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
void strncat_overflow_0(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strncat(x, y, strlen(y));
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}

void strncat_overflow_1(char *y) {
  char x[4] = "12";
  if (strlen(y) == 3)
    strncat(x, y, strlen(y));
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}

void strncat_overflow_2(char *y) {
  char x[4] = "12";
  if (strlen(y) == 2)
    strncat(x, y, strlen(y));
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}

void strncat_overflow_3(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strncat(x, y, 2);
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}
#endif

void strncat_no_overflow_1(char *y) {
  char x[5] = "12";
  if (strlen(y) == 2)
    strncat(x, y, strlen(y)); // no-warning
}

void strncat_no_overflow_2(char *y) {
  char x[4] = "12";
  if (strlen(y) == 4)
    strncat(x, y, 1); // no-warning
}

void strncat_symbolic_dst_length(char *dst) {
  strncat(dst, "1234", 5);
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
void strncat_symbolic_src_length(char *src) {
  char dst[8] = "1234";
  strncat(dst, src, 3);
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}

  char dst2[8] = "1234";
  strncat(dst2, src, 4);
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}

void strncat_unknown_src_length(char *src, int offset) {
  char dst[8] = "1234";
  strncat(dst, &src[offset], 3);
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}

  char dst2[8] = "1234";
  strncat(dst2, &src[offset], 4);
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}
#endif

// There is no strncat_unknown_dst_length because if we can't get a symbolic
// length for the "before" strlen, we won't be able to set one for "after".

void strncat_symbolic_limit(unsigned limit) {
  char dst[6] = "1234";
  char src[] = "567";
  strncat(dst, src, limit); // no-warning

  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(dst) == 4); // expected-warning{{UNKNOWN}}
}

void strncat_unknown_limit(float limit) {
  char dst[6] = "1234";
  char src[] = "567";
  strncat(dst, src, (size_t)limit); // no-warning

  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(dst) == 4); // expected-warning{{UNKNOWN}}
}

void strncat_too_big(char *dst, char *src) {
  // We assume this will never actually happen, so we don't get a warning.
  if (strlen(dst) != (((size_t)0) - 2))
    return;
  if (strlen(src) != 2)
    return;
  strncat(dst, src, 2);
}

void strncat_zero(char *src) {
  char dst[] = "123";
  strncat(dst, src, 0); // no-warning
}

void strncat_empty(void) {
  char dst[8] = "123";
  char src[] = "";
  strncat(dst, src, 4); // no-warning
}

//===----------------------------------------------------------------------===
// strcmp()
//===----------------------------------------------------------------------===

#define strcmp BUILTIN(strcmp)
int strcmp(const char * s1, const char * s2);

void strcmp_check_modelling(void) {
  char *x = "aa";
  char *y = "a";
  clang_analyzer_eval(strcmp(x, y) > 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strcmp(x, y) <= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strcmp(x, y) > 1); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(strcmp(y, x) < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strcmp(y, x) >= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strcmp(y, x) < -1); // expected-warning{{UNKNOWN}}
}

void strcmp_constant0(void) {
  clang_analyzer_eval(strcmp("123", "123") == 0); // expected-warning{{TRUE}}
}

void strcmp_constant_and_var_0(void) {
  char *x = "123";
  clang_analyzer_eval(strcmp(x, "123") == 0); // expected-warning{{TRUE}}
}

void strcmp_constant_and_var_1(void) {
  char *x = "123";
  clang_analyzer_eval(strcmp("123", x) == 0); // expected-warning{{TRUE}}
}

void strcmp_0(void) {
  char *x = "123";
  char *y = "123";
  clang_analyzer_eval(strcmp(x, y) == 0); // expected-warning{{TRUE}}
}

void strcmp_1(void) {
  char *x = "234";
  char *y = "123";
  clang_analyzer_eval(strcmp(x, y) > 0); // expected-warning{{TRUE}}
}

void strcmp_2(void) {
  char *x = "123";
  char *y = "234";
  clang_analyzer_eval(strcmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcmp_null_0(void) {
  char *x = NULL;
  char *y = "123";
  strcmp(x, y); // expected-warning{{Null pointer passed as 1st argument to string comparison function}}
}

void strcmp_null_1(void) {
  char *x = "123";
  char *y = NULL;
  strcmp(x, y); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

void strcmp_diff_length_0(void) {
  char *x = "12345";
  char *y = "234";
  clang_analyzer_eval(strcmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcmp_diff_length_1(void) {
  char *x = "123";
  char *y = "23456";
  clang_analyzer_eval(strcmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcmp_diff_length_2(void) {
  char *x = "12345";
  char *y = "123";
  clang_analyzer_eval(strcmp(x, y) > 0); // expected-warning{{TRUE}}
}

void strcmp_diff_length_3(void) {
  char *x = "123";
  char *y = "12345";
  clang_analyzer_eval(strcmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcmp_embedded_null (void) {
	clang_analyzer_eval(strcmp("\0z", "\0y") == 0); // expected-warning{{TRUE}}
}

void strcmp_unknown_arg (char *unknown) {
	clang_analyzer_eval(strcmp(unknown, unknown) == 0); // expected-warning{{TRUE}}
}

union argument {
   char *f;
};

void function_pointer_cast_helper(char **a) {
  strcmp("Hi", *a); // PR24951 crash
}

void strcmp_union_function_pointer_cast(union argument a) {
  void (*fPtr)(union argument *) = (void (*)(union argument *))function_pointer_cast_helper;

  fPtr(&a);
}

int strcmp_null_argument(char *a) {
  char *b = 0;
  // Do not warn about the first argument!
  return strcmp(a, b); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

//===----------------------------------------------------------------------===
// strncmp()
//===----------------------------------------------------------------------===

#define strncmp BUILTIN(strncmp)
int strncmp(const char *s1, const char *s2, size_t n);

void strncmp_check_modelling(void) {
  char *x = "aa";
  char *y = "a";
  clang_analyzer_eval(strncmp(x, y, 2) > 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strncmp(x, y, 2) <= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strncmp(x, y, 2) > 1); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(strncmp(y, x, 2) < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strncmp(y, x, 2) >= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strncmp(y, x, 2) < -1); // expected-warning{{UNKNOWN}}
}

void strncmp_constant0(void) {
  clang_analyzer_eval(strncmp("123", "123", 3) == 0); // expected-warning{{TRUE}}
}

void strncmp_constant_and_var_0(void) {
  char *x = "123";
  clang_analyzer_eval(strncmp(x, "123", 3) == 0); // expected-warning{{TRUE}}
}

void strncmp_constant_and_var_1(void) {
  char *x = "123";
  clang_analyzer_eval(strncmp("123", x, 3) == 0); // expected-warning{{TRUE}}
}

void strncmp_0(void) {
  char *x = "123";
  char *y = "123";
  clang_analyzer_eval(strncmp(x, y, 3) == 0); // expected-warning{{TRUE}}
}

void strncmp_1(void) {
  char *x = "234";
  char *y = "123";
  clang_analyzer_eval(strncmp(x, y, 3) > 0); // expected-warning{{TRUE}}
}

void strncmp_2(void) {
  char *x = "123";
  char *y = "234";
  clang_analyzer_eval(strncmp(x, y, 3) < 0); // expected-warning{{TRUE}}
}

void strncmp_null_0(void) {
  char *x = NULL;
  char *y = "123";
  strncmp(x, y, 3); // expected-warning{{Null pointer passed as 1st argument to string comparison function}}
}

void strncmp_null_1(void) {
  char *x = "123";
  char *y = NULL;
  strncmp(x, y, 3); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

void strncmp_diff_length_0(void) {
  char *x = "12345";
  char *y = "234";
  clang_analyzer_eval(strncmp(x, y, 5) < 0); // expected-warning{{TRUE}}
}

void strncmp_diff_length_1(void) {
  char *x = "123";
  char *y = "23456";
  clang_analyzer_eval(strncmp(x, y, 5) < 0); // expected-warning{{TRUE}}
}

void strncmp_diff_length_2(void) {
  char *x = "12345";
  char *y = "123";
  clang_analyzer_eval(strncmp(x, y, 5) > 0); // expected-warning{{TRUE}}
}

void strncmp_diff_length_3(void) {
  char *x = "123";
  char *y = "12345";
  clang_analyzer_eval(strncmp(x, y, 5) < 0); // expected-warning{{TRUE}}
}

void strncmp_diff_length_4(void) {
  char *x = "123";
  char *y = "12345";
  clang_analyzer_eval(strncmp(x, y, 3) == 0); // expected-warning{{TRUE}}
}

void strncmp_diff_length_5(void) {
  char *x = "012";
  char *y = "12345";
  clang_analyzer_eval(strncmp(x, y, 3) < 0); // expected-warning{{TRUE}}
}

void strncmp_diff_length_6(void) {
  char *x = "234";
  char *y = "12345";
  clang_analyzer_eval(strncmp(x, y, 3) > 0); // expected-warning{{TRUE}}
}

void strncmp_embedded_null (void) {
	clang_analyzer_eval(strncmp("ab\0zz", "ab\0yy", 4) == 0); // expected-warning{{TRUE}}
}

int strncmp_null_argument(char *a, size_t n) {
  char *b = 0;
  // Do not warn about the first argument!
  return strncmp(a, b, n); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

//===----------------------------------------------------------------------===
// strcasecmp()
//===----------------------------------------------------------------------===

#define strcasecmp BUILTIN(strcasecmp)
int strcasecmp(const char *s1, const char *s2);

void strcasecmp_check_modelling(void) {
  char *x = "aa";
  char *y = "a";
  clang_analyzer_eval(strcasecmp(x, y) > 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strcasecmp(x, y) <= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strcasecmp(x, y) > 1); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(strcasecmp(y, x) < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strcasecmp(y, x) >= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strcasecmp(y, x) < -1); // expected-warning{{UNKNOWN}}
}

void strcasecmp_constant0(void) {
  clang_analyzer_eval(strcasecmp("abc", "Abc") == 0); // expected-warning{{TRUE}}
}

void strcasecmp_constant_and_var_0(void) {
  char *x = "abc";
  clang_analyzer_eval(strcasecmp(x, "Abc") == 0); // expected-warning{{TRUE}}
}

void strcasecmp_constant_and_var_1(void) {
  char *x = "abc";
  clang_analyzer_eval(strcasecmp("Abc", x) == 0); // expected-warning{{TRUE}}
}

void strcasecmp_0(void) {
  char *x = "abc";
  char *y = "Abc";
  clang_analyzer_eval(strcasecmp(x, y) == 0); // expected-warning{{TRUE}}
}

void strcasecmp_1(void) {
  char *x = "Bcd";
  char *y = "abc";
  clang_analyzer_eval(strcasecmp(x, y) > 0); // expected-warning{{TRUE}}
}

void strcasecmp_2(void) {
  char *x = "abc";
  char *y = "Bcd";
  clang_analyzer_eval(strcasecmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcasecmp_null_0(void) {
  char *x = NULL;
  char *y = "123";
  strcasecmp(x, y); // expected-warning{{Null pointer passed as 1st argument to string comparison function}}
}

void strcasecmp_null_1(void) {
  char *x = "123";
  char *y = NULL;
  strcasecmp(x, y); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

void strcasecmp_diff_length_0(void) {
  char *x = "abcde";
  char *y = "aBd";
  clang_analyzer_eval(strcasecmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcasecmp_diff_length_1(void) {
  char *x = "abc";
  char *y = "aBdef";
  clang_analyzer_eval(strcasecmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcasecmp_diff_length_2(void) {
  char *x = "aBcDe";
  char *y = "abc";
  clang_analyzer_eval(strcasecmp(x, y) > 0); // expected-warning{{TRUE}}
}

void strcasecmp_diff_length_3(void) {
  char *x = "aBc";
  char *y = "abcde";
  clang_analyzer_eval(strcasecmp(x, y) < 0); // expected-warning{{TRUE}}
}

void strcasecmp_embedded_null (void) {
	clang_analyzer_eval(strcasecmp("ab\0zz", "ab\0yy") == 0); // expected-warning{{TRUE}}
}

int strcasecmp_null_argument(char *a) {
  char *b = 0;
  // Do not warn about the first argument!
  return strcasecmp(a, b); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

//===----------------------------------------------------------------------===
// strncasecmp()
//===----------------------------------------------------------------------===

#define strncasecmp BUILTIN(strncasecmp)
int strncasecmp(const char *s1, const char *s2, size_t n);

void strncasecmp_check_modelling(void) {
  char *x = "aa";
  char *y = "a";
  clang_analyzer_eval(strncasecmp(x, y, 2) > 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strncasecmp(x, y, 2) <= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strncasecmp(x, y, 2) > 1); // expected-warning{{UNKNOWN}}

  clang_analyzer_eval(strncasecmp(y, x, 2) < 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strncasecmp(y, x, 2) >= 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(strncasecmp(y, x, 2) < -1); // expected-warning{{UNKNOWN}}
}

void strncasecmp_constant0(void) {
  clang_analyzer_eval(strncasecmp("abc", "Abc", 3) == 0); // expected-warning{{TRUE}}
}

void strncasecmp_constant_and_var_0(void) {
  char *x = "abc";
  clang_analyzer_eval(strncasecmp(x, "Abc", 3) == 0); // expected-warning{{TRUE}}
}

void strncasecmp_constant_and_var_1(void) {
  char *x = "abc";
  clang_analyzer_eval(strncasecmp("Abc", x, 3) == 0); // expected-warning{{TRUE}}
}

void strncasecmp_0(void) {
  char *x = "abc";
  char *y = "Abc";
  clang_analyzer_eval(strncasecmp(x, y, 3) == 0); // expected-warning{{TRUE}}
}

void strncasecmp_1(void) {
  char *x = "Bcd";
  char *y = "abc";
  clang_analyzer_eval(strncasecmp(x, y, 3) > 0); // expected-warning{{TRUE}}
}

void strncasecmp_2(void) {
  char *x = "abc";
  char *y = "Bcd";
  clang_analyzer_eval(strncasecmp(x, y, 3) < 0); // expected-warning{{TRUE}}
}

void strncasecmp_null_0(void) {
  char *x = NULL;
  char *y = "123";
  strncasecmp(x, y, 3); // expected-warning{{Null pointer passed as 1st argument to string comparison function}}
}

void strncasecmp_null_1(void) {
  char *x = "123";
  char *y = NULL;
  strncasecmp(x, y, 3); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

void strncasecmp_diff_length_0(void) {
  char *x = "abcde";
  char *y = "aBd";
  clang_analyzer_eval(strncasecmp(x, y, 5) < 0); // expected-warning{{TRUE}}
}

void strncasecmp_diff_length_1(void) {
  char *x = "abc";
  char *y = "aBdef";
  clang_analyzer_eval(strncasecmp(x, y, 5) < 0); // expected-warning{{TRUE}}
}

void strncasecmp_diff_length_2(void) {
  char *x = "aBcDe";
  char *y = "abc";
  clang_analyzer_eval(strncasecmp(x, y, 5) > 0); // expected-warning{{TRUE}}
}

void strncasecmp_diff_length_3(void) {
  char *x = "aBc";
  char *y = "abcde";
  clang_analyzer_eval(strncasecmp(x, y, 5) < 0); // expected-warning{{TRUE}}
}

void strncasecmp_diff_length_4(void) {
  char *x = "abcde";
  char *y = "aBc";
  clang_analyzer_eval(strncasecmp(x, y, 3) == 0); // expected-warning{{TRUE}}
}

void strncasecmp_diff_length_5(void) {
  char *x = "abcde";
  char *y = "aBd";
  clang_analyzer_eval(strncasecmp(x, y, 3) < 0); // expected-warning{{TRUE}}
}

void strncasecmp_diff_length_6(void) {
  char *x = "aBDe";
  char *y = "abc";
  clang_analyzer_eval(strncasecmp(x, y, 3) > 0); // expected-warning{{TRUE}}
}

void strncasecmp_embedded_null (void) {
	clang_analyzer_eval(strncasecmp("ab\0zz", "ab\0yy", 4) == 0); // expected-warning{{TRUE}}
}

int strncasecmp_null_argument(char *a, size_t n) {
  char *b = 0;
  // Do not warn about the first argument!
  return strncasecmp(a, b, n); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

//===----------------------------------------------------------------------===
// strsep()
//===----------------------------------------------------------------------===

char *strsep(char **stringp, const char *delim);

void strsep_null_delim(char *s) {
  strsep(&s, NULL); // expected-warning{{Null pointer passed as 2nd argument to strsep()}}
}

void strsep_null_search(void) {
  strsep(NULL, ""); // expected-warning{{Null pointer passed as 1st argument to strsep()}}
}

void strsep_return_original_pointer(char *s) {
  char *original = s;
  char *result = strsep(&s, ""); // no-warning
  clang_analyzer_eval(original == result); // expected-warning{{TRUE}}
}

void strsep_null_string(void) {
  char *s = NULL;
  char *result = strsep(&s, ""); // no-warning
  clang_analyzer_eval(result == NULL); // expected-warning{{TRUE}}
}

void strsep_changes_input_pointer(char *s) {
  char *original = s;
  strsep(&s, ""); // no-warning
  clang_analyzer_eval(s == original); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(s == NULL); // expected-warning{{UNKNOWN}}

  // Check that the value is symbolic.
  if (s == NULL) {
    clang_analyzer_eval(s == NULL); // expected-warning{{TRUE}}
  }
}

void strsep_changes_input_string(void) {
  char str[] = "abc";

  clang_analyzer_eval(str[1] == 'b'); // expected-warning{{TRUE}}

  char *s = str;
  strsep(&s, "b"); // no-warning

  // The real strsep will change the first delimiter it finds into a NUL
  // character. For now, we just model the invalidation.
  clang_analyzer_eval(str[1] == 'b'); // expected-warning{{UNKNOWN}}
}

//===----------------------------------------------------------------------===
// memset() / explicit_bzero() / bzero()
//===----------------------------------------------------------------------===

void *memset(void *dest, int ch, size_t count);

void bzero(void *dst, size_t count);
void explicit_bzero(void *dest, size_t count);

void *malloc(size_t size);
void free(void *);

void memset1_char_array_null(void) {
  char str[] = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  memset(str, '\0', 2);
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{TRUE}}
}

void memset2_char_array_null(void) {
  char str[] = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  memset(str, '\0', strlen(str) + 1);
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(str[2] == 0);      // expected-warning{{TRUE}}
}

void memset3_char_malloc_null(void) {
  char *str = (char *)malloc(10 * sizeof(char));
  memset(str + 1, '\0', 8);
  clang_analyzer_eval(str[1] == 0); // expected-warning{{UNKNOWN}}
  free(str);
}

void memset4_char_malloc_null(void) {
  char *str = (char *)malloc(10 * sizeof(char));
  //void *str = malloc(10 * sizeof(char));
  memset(str, '\0', 10);
  clang_analyzer_eval(str[1] == 0);      // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{TRUE}}
  free(str);
}

#ifdef SUPPRESS_OUT_OF_BOUND
void memset5_char_malloc_overflow_null(void) {
  char *str = (char *)malloc(10 * sizeof(char));
  memset(str, '\0', 12);
  clang_analyzer_eval(str[1] == 0); // expected-warning{{UNKNOWN}}
  free(str);
}
#endif

void memset6_char_array_nonnull(void) {
  char str[] = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  memset(str, '0', 2);
  clang_analyzer_eval(str[0] == 'a');    // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{UNKNOWN}}
}

#ifdef SUPPRESS_OUT_OF_BOUND
void memset8_char_array_nonnull(void) {
  char str[5] = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  memset(str, '0', 10); // expected-warning{{'memset' will always overflow; destination buffer has size 5, but size argument is 10}}
  clang_analyzer_eval(str[0] != '0');     // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen(str) >= 10); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(str) < 10);  // expected-warning{{FALSE}}
}
#endif

struct POD_memset {
  int num;
  char c;
};

void memset10_struct(void) {
  struct POD_memset pod;
  char *str = (char *)&pod;
  pod.num = 1;
  pod.c = 1;
  clang_analyzer_eval(pod.num == 0); // expected-warning{{FALSE}}
  memset(str, 0, sizeof(struct POD_memset));
  clang_analyzer_eval(pod.num == 0); // expected-warning{{TRUE}}
}

#ifdef SUPPRESS_OUT_OF_BOUND
void memset11_struct_field(void) {
  struct POD_memset pod;
  pod.num = 1;
  pod.c = '1';
  memset(&pod.num, 0, sizeof(struct POD_memset));

  clang_analyzer_eval(pod.num == 0);  // expected-warning{{TRUE}}
  clang_analyzer_eval(pod.c == '\0'); // expected-warning{{TRUE}}
}

void memset12_struct_field(void) {
  struct POD_memset pod;
  pod.num = 1;
  pod.c = '1';
  memset(&pod.c, 0, sizeof(struct POD_memset)); // expected-warning {{'memset' will always overflow; destination buffer has size 4, but size argument is 8}}
  clang_analyzer_eval(pod.num == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(pod.c == 0);   // expected-warning{{UNKNOWN}}
}

union U_memset {
  int i;
  double d;
  char c;
};

void memset13_union_field(void) {
  union U_memset u;
  u.i = 5;
  memset(&u.i, '\0', sizeof(union U_memset));
  // Note: This should be TRUE, analyzer can't handle union perfectly now.
  clang_analyzer_eval(u.d == 0); // expected-warning{{UNKNOWN}}
}
#endif

void memset14_region_cast(void) {
  char *str = (char *)malloc(10 * sizeof(int));
  int *array = (int *)str;
  memset(array, 0, 10 * sizeof(int));
  clang_analyzer_eval(str[10] == '\0');            // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen((char *)array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(str) == 0);           // expected-warning{{TRUE}}
  free(str);
}

void memset15_region_cast(void) {
  char *str = (char *)malloc(10 * sizeof(int));
  int *array = (int *)str;
  memset(array, 0, 5 * sizeof(int));
  clang_analyzer_eval(str[10] == '\0');            // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen((char *)array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(str) == 0);           // expected-warning{{TRUE}}
  free(str);
}

int memset20_scalar(void) {
  int *x = malloc(sizeof(int));
  *x = 10;
  memset(x, 0, sizeof(int));
  int num = 1 / *x; // expected-warning{{Division by zero}}
  free(x);
  return num;
}

int memset21_scalar(void) {
  int *x = malloc(sizeof(int));
  memset(x, 0, 1);
  int num = 1 / *x;
  free(x);
  return num;
}

void memset22_array(void) {
  int array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  clang_analyzer_eval(array[1] == 2); // expected-warning{{TRUE}}
  memset(array, 0, sizeof(array));
  clang_analyzer_eval(array[1] == 0); // expected-warning{{TRUE}}
}

void memset23_array_pod_object(void) {
  struct POD_memset array[10];
  array[1].num = 10;
  array[1].c = 'c';
  clang_analyzer_eval(array[1].num == 10); // expected-warning{{TRUE}}
  memset(&array[1], 0, sizeof(struct POD_memset));
  clang_analyzer_eval(array[1].num == 0); // expected-warning{{UNKNOWN}}
}

void memset24_array_pod_object(void) {
  struct POD_memset array[10];
  array[1].num = 10;
  array[1].c = 'c';
  clang_analyzer_eval(array[1].num == 10); // expected-warning{{TRUE}}
  memset(array, 0, sizeof(array));
  clang_analyzer_eval(array[1].num == 0); // expected-warning{{TRUE}}
}

void memset25_symbol(char c) {
  char array[10] = {1};
  if (c != 0)
    return;

  memset(array, c, 10);

  clang_analyzer_eval(strlen(array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[4] == 0); // expected-warning{{TRUE}}
}

void memset26_upper_UCHAR_MAX(void) {
  char array[10] = {1};

  memset(array, 1024, 10);

  clang_analyzer_eval(strlen(array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[4] == 0); // expected-warning{{TRUE}}
}

void bzero1_null(void) {
  char *a = NULL;

  bzero(a, 10); // expected-warning{{Null pointer passed as 1st argument to memory clearance function}}
}

void bzero2_char_array_null(void) {
  char str[] = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  bzero(str, 2);
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{TRUE}}
}

void bzero3_char_ptr_null(void) {
  char *str = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  bzero(str + 2, 2);
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{FALSE}}
}

void explicit_bzero1_null(void) {
  char *a = NULL;

  explicit_bzero(a, 10); // expected-warning{{Null pointer passed as 1st argument to memory clearance function}}
}

void explicit_bzero2_clear_mypassword(void) {
  char passwd[7] = "passwd";

  explicit_bzero(passwd, sizeof(passwd)); // no-warning

  clang_analyzer_eval(strlen(passwd) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(passwd[0] == '\0'); // expected-warning{{TRUE}}
}

void explicit_bzero3_out_ofbound(void) {
  char *privkey = (char *)malloc(7);
  const char newprivkey[10] = "mysafekey";

  strcpy(privkey, "random");
  explicit_bzero(privkey, sizeof(newprivkey));
#ifndef SUPPRESS_OUT_OF_BOUND
  // expected-warning@-2 {{Memory clearance function overflows the destination buffer}}
#endif
  clang_analyzer_eval(privkey[0] == '\0');
#ifdef SUPPRESS_OUT_OF_BOUND
  // expected-warning@-2 {{UNKNOWN}}
#endif
  free(privkey);
}

//===----------------------------------------------------------------------===
// FIXMEs
//===----------------------------------------------------------------------===

// The analyzer_eval call below should evaluate to true. We are being too
// aggressive in marking the (length of) src symbol dead. The length of dst
// depends on src. This could be explicitly specified in the checker or the
// logic for handling MetadataSymbol in SymbolManager needs to change.
void strcat_symbolic_src_length(char *src) {
	char dst[8] = "1234";
	strcat(dst, src);
  clang_analyzer_eval(strlen(dst) >= 4); // expected-warning{{UNKNOWN}}
}


// The analyzer_eval call below should evaluate to true. Most likely the same
// issue as the test above.
void strncpy_exactly_matching_buffer2(char *y) {
	if (strlen(y) >= 4)
		return;

	char x[4];
	strncpy(x, y, 4); // no-warning

	// This time, we know that y fits in x anyway.
  clang_analyzer_eval(strlen(x) <= 3); // expected-warning{{UNKNOWN}}
}

void memset7_char_array_nonnull(void) {
  char str[5] = "abcd";
  clang_analyzer_eval(strlen(str) == 4); // expected-warning{{TRUE}}
  memset(str, '0', 5);
  // FIXME: This should be TRUE.
  clang_analyzer_eval(str[0] == '0');    // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen(str) >= 5); // expected-warning{{TRUE}}
}

void memset16_region_cast(void) {
  char *str = (char *)malloc(10 * sizeof(int));
  int *array = (int *)str;
  memset(array, '0', 10 * sizeof(int));
  // FIXME: This should be TRUE.
  clang_analyzer_eval(str[10] == '0');                            // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen((char *)array) >= 10 * sizeof(int)); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(str) >= 10 * sizeof(int));           // expected-warning{{TRUE}}
  free(str);
}

#ifdef SUPPRESS_OUT_OF_BOUND
void memset17_region_cast(void) {
  char *str = (char *)malloc(10 * sizeof(int));
  int *array = (int *)str;
  memset(array, '0', 12 * sizeof(int));
  clang_analyzer_eval(str[10] == '0');                            // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen((char *)array) >= 12 * sizeof(int)); // expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(str) >= 12 * sizeof(int));           // expected-warning{{TRUE}}
  free(str);
}

void memset18_memset_multiple_times(void) {
  char *str = (char *)malloc(10 * sizeof(char));
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{UNKNOWN}}

  memset(str + 2, '\0', 10 * sizeof(char));
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(str[1] == '\0');   // expected-warning{{UNKNOWN}}

  memset(str, '0', 10 * sizeof(char));
  clang_analyzer_eval(strlen(str) >= 10); // expected-warning{{TRUE}}
  // FIXME: This should be TRUE.
  clang_analyzer_eval(str[1] == '0');     // expected-warning{{UNKNOWN}}

  free(str);
}

void memset19_memset_multiple_times(void) {
  char *str = (char *)malloc(10 * sizeof(char));
  clang_analyzer_eval(strlen(str) == 0); // expected-warning{{UNKNOWN}}

  memset(str, '0', 10 * sizeof(char));
  clang_analyzer_eval(strlen(str) >= 10); // expected-warning{{TRUE}}
  // FIXME: This should be TRUE.
  clang_analyzer_eval(str[1] == '0');     // expected-warning{{UNKNOWN}}

  memset(str + 2, '\0', 10 * sizeof(char));
  clang_analyzer_eval(strlen(str) >= 10); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(str[1] == '0');     // expected-warning{{UNKNOWN}}

  free(str);
}
#endif

// The analyzer does not support binding a symbol with default binding.
void memset27_symbol(char c) {
  char array[10] = {0};
  if (c < 10)
    return;

  memset(array, c, 10);

  clang_analyzer_eval(strlen(array) >= 10); // expected-warning{{TRUE}}
  // FIXME: This should be TRUE.
  clang_analyzer_eval(array[4] >= 10); // expected-warning{{UNKNOWN}}
}

void memset28(void) {
  short x;
  memset(&x, 1, sizeof(short));
  // This should be true.
  clang_analyzer_eval(x == 0x101); // expected-warning{{UNKNOWN}}
}

void memset29_plain_int_zero(void) {
  short x;
  memset(&x, 0, sizeof(short));
  clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
}

void test_memset_chk(void) {
  int x;
  __builtin___memset_chk(&x, 0, sizeof(x), __builtin_object_size(&x, 0));
  clang_analyzer_eval(x == 0); // expected-warning{{TRUE}}
}

#ifndef SUPPRESS_OUT_OF_BOUND
void strcpy_no_overflow_2(char *y) {
  char x[3];
  // FIXME: string literal modeling does not account for embedded NULLs.
  //        This case should not elicit a warning, but does.
  //        See discussion at https://reviews.llvm.org/D129269
  strcpy(x, "12\0"); // expected-warning{{String copy function overflows the destination buffer}}
}
#else
void strcpy_no_overflow_2(char *y) {
  char x[3];
  strcpy(x, "12\0");
}
#endif

#ifndef SUPPRESS_OUT_OF_BOUND
void testStrcpyDestinationWritableFirstByte(void) {
  char dst[10];
  char *p = dst - 8;
  strcpy(p, "src"); // expected-warning {{String copy function overflows the destination buffer}}
}

void CWE124_Buffer_Underwrite__malloc_char_cpy() {
  char * dataBuffer = (char *)malloc(100*sizeof(char));
  if (dataBuffer == NULL) return;
  memset(dataBuffer, 'A', 100-1);
  dataBuffer[100-1] = '\0';
  char * data = dataBuffer - 8;
  char source[100];
  memset(source, 'C', 100-1); // fill with 'C's
  source[100-1] = '\0'; // null terminate
  strcpy(data, source); // expected-warning {{String copy function overflows the destination buffer}}
  free(dataBuffer);
}
#endif

#ifndef SUPPRESS_OUT_OF_BOUND
void testStrncpyDestinationWritableFirstByte(void) {
  char source[100];
  use_string(source); // escape
  char buf[100];
  char *p = buf - 8;
  strncpy(p, source, 100-1); // expected-warning {{String copy function overflows the destination buffer}}
}

void CWE124_Buffer_Underwrite__malloc_char_ncpy() {
  char * dataBuffer = (char *)malloc(100*sizeof(char));
  if (dataBuffer == 0) return;
  memset(dataBuffer, 'A', 100-1);
  dataBuffer[100-1] = '\0';
  char *data = dataBuffer - 8;

  char source[100];
  memset(source, 'C', 100-1); // fill with 'C's
  source[100-1] = '\0'; // null terminate
  strncpy(data, source, 100-1); // expected-warning {{String copy function overflows the destination buffer}}
  data[100-1] = '\0'; // null terminate
  free(dataBuffer);
}
#endif

#ifndef SUPPRESS_OUT_OF_BOUND
void CWE124_Buffer_Underwrite__malloc_char_memcpy() {
  char * dataBuffer = (char *)malloc(100*sizeof(char));
  if (dataBuffer == NULL) return;
  memset(dataBuffer, 'A', 100-1);
  dataBuffer[100-1] = '\0';
  char *data = dataBuffer - 8;

  char source[100];
  memset(source, 'C', 100-1); // fill with 'C's
  source[100-1] = '\0'; // null terminate

  memcpy(data, source, 100*sizeof(char)); // expected-warning {{Memory copy function overflows the destination buffer}}
  data[100-1] = '\0';
  free(dataBuffer);
}
#endif
