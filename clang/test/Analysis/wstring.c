// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-disable-checker=alpha.unix.cstring.UninitializedRead \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false  
//
// RUN: %clang_analyze_cc1 -verify %s -DUSE_BUILTINS \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-disable-checker=alpha.unix.cstring.UninitializedRead \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

//===----------------------------------------------------------------------===
// Declarations
//===----------------------------------------------------------------------===

// Some functions are implemented as builtins. These should be #defined as
// BUILTIN(f), which will prepend "__builtin_" if USE_BUILTINS is defined.

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else /* USE_BUILTINS */
# define BUILTIN(f) f
#endif /* USE_BUILTINS */

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

void clang_analyzer_eval(int);

//===----------------------------------------------------------------------===
// wmemcpy()
//===----------------------------------------------------------------------===

#define wmemcpy BUILTIN(wmemcpy)
wchar_t *wmemcpy(wchar_t *restrict s1, const wchar_t *restrict s2, size_t n);

void wmemcpy0 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[4] = {0};

  wmemcpy(dst, src, 4); // no-warning

  clang_analyzer_eval(wmemcpy(dst, src, 4) == dst); // expected-warning{{TRUE}}

  // If we actually model the copy, we can make this known.
  // The important thing for now is that the old value has been invalidated.
  clang_analyzer_eval(dst[0] != 0); // expected-warning{{UNKNOWN}}
}

void wmemcpy1 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[10];

  wmemcpy(dst, src, 5); // expected-warning{{Memory copy function accesses out-of-bound array element}}
}

void wmemcpy2 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[1];

  wmemcpy(dst, src, 4); // expected-warning {{Memory copy function overflows the destination buffer}}
}

void wmemcpy3 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[3];

  wmemcpy(dst+1, src+2, 2); // no-warning
}

void wmemcpy4 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[10];

  wmemcpy(dst+2, src+2, 3); // expected-warning{{Memory copy function accesses out-of-bound array element}}
}

void wmemcpy5(void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[3];

  wmemcpy(dst + 2, src + 2, 2); // expected-warning{{Memory copy function overflows the destination buffer}}
}

void wmemcpy6(void) {
  wchar_t a[4] = {0};
  wmemcpy(a, a, 2); // expected-warning{{overlapping}}
}

void wmemcpy7(void) {
  wchar_t a[4] = {0};
  wmemcpy(a+2, a+1, 2); // expected-warning{{overlapping}}
}

void wmemcpy8(void) {
  wchar_t a[4] = {0};
  wmemcpy(a+1, a+2, 2); // expected-warning{{overlapping}}
}

void wmemcpy9(void) {
  wchar_t a[4] = {0};
  wmemcpy(a+2, a+1, 1); // no-warning
  wmemcpy(a+1, a+2, 1); // no-warning
}

void wmemcpy10(void) {
  wchar_t a[4] = {0};
  wmemcpy(0, a, 1); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
}

void wmemcpy11(void) {
  wchar_t a[4] = {0};
  wmemcpy(a, 0, 1); // expected-warning{{Null pointer passed as 2nd argument to memory copy function}}
}

void wmemcpy12(void) {
  wchar_t a[4] = {0};
  wmemcpy(0, a, 0); // no-warning
}

void wmemcpy13(void) {
  wchar_t a[4] = {0};
  wmemcpy(a, 0, 0); // no-warning
}

void wmemcpy_unknown_size (size_t n) {
  wchar_t a[4], b[4] = {1};
  clang_analyzer_eval(wmemcpy(a, b, n) == a); // expected-warning{{TRUE}}
}

void wmemcpy_unknown_size_warn (size_t n) {
  wchar_t a[4];
  void *result = wmemcpy(a, 0, n); // expected-warning{{Null pointer passed as 2nd argument to memory copy function}}
  clang_analyzer_eval(result == a); // no-warning (above is fatal)
}

//===----------------------------------------------------------------------===
// wmempcpy()
//===----------------------------------------------------------------------===

wchar_t *wmempcpy(wchar_t *restrict s1, const wchar_t *restrict s2, size_t n);

void wmempcpy0 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[5] = {0};

  wmempcpy(dst, src, 4); // no-warning

  clang_analyzer_eval(wmempcpy(dst, src, 4) == &dst[4]); // expected-warning{{TRUE}}

  // If we actually model the copy, we can make this known.
  // The important thing for now is that the old value has been invalidated.
  clang_analyzer_eval(dst[0] != 0); // expected-warning{{UNKNOWN}}
}

void wmempcpy1 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[10];

  wmempcpy(dst, src, 5); // expected-warning{{Memory copy function accesses out-of-bound array element}}
}

void wmempcpy2 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[1];

  wmempcpy(dst, src, 4); // expected-warning{{Memory copy function overflows the destination buffer}}
}

void wmempcpy3 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[3];

  wmempcpy(dst+1, src+2, 2); // no-warning
}

void wmempcpy4 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[10];

  wmempcpy(dst+2, src+2, 3); // expected-warning{{Memory copy function accesses out-of-bound array element}}
}

void wmempcpy5(void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[3];

  wmempcpy(dst + 2, src + 2, 2); // expected-warning{{Memory copy function overflows the destination buffer}}
}

void wmempcpy6(void) {
  wchar_t a[4] = {0};
  wmempcpy(a, a, 2); // expected-warning{{overlapping}}
}

void wmempcpy7(void) {
  wchar_t a[4] = {0};
  wmempcpy(a+2, a+1, 2); // expected-warning{{overlapping}}
}

void wmempcpy8(void) {
  wchar_t a[4] = {0};
  wmempcpy(a+1, a+2, 2); // expected-warning{{overlapping}}
}

void wmempcpy9(void) {
  wchar_t a[4] = {0};
  wmempcpy(a+2, a+1, 1); // no-warning
  wmempcpy(a+1, a+2, 1); // no-warning
}

void wmempcpy10(void) {
  wchar_t a[4] = {0};
  wmempcpy(0, a, 1); // expected-warning{{Null pointer passed as 1st argument to memory copy function}}
}

void wmempcpy11(void) {
  wchar_t a[4] = {0};
  wmempcpy(a, 0, 1); // expected-warning{{Null pointer passed as 2nd argument to memory copy function}}
}

void wmempcpy12(void) {
  wchar_t a[4] = {0};
  wmempcpy(0, a, 0); // no-warning
}

void wmempcpy13(void) {
  wchar_t a[4] = {0};
  wmempcpy(a, 0, 0); // no-warning
}

void wmempcpy14(void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[5] = {0};
  wchar_t *p;

  p = wmempcpy(dst, src, 4);

  clang_analyzer_eval(p == &dst[4]); // expected-warning{{TRUE}}
}

struct st {
  wchar_t i;
  wchar_t j;
};

void wmempcpy15(void) {
  struct st s1 = {0};
  struct st s2;
  struct st *p1;
  struct st *p2;

  p1 = (&s2) + 1;
  p2 = (struct st *)wmempcpy((wchar_t *)&s2, (wchar_t *)&s1, 2);

  clang_analyzer_eval(p1 == p2); // expected-warning{{TRUE}}
}

void wmempcpy16(void) {
  struct st s1[10] = {{0}};
  struct st s2[10];
  struct st *p1;
  struct st *p2;

  p1 = (&s2[0]) + 5;
  p2 = (struct st *)wmempcpy((wchar_t *)&s2[0], (wchar_t *)&s1[0], 5 * 2);

  clang_analyzer_eval(p1 == p2); // expected-warning{{TRUE}}
}

void wmempcpy_unknown_size_warn (size_t n) {
  wchar_t a[4];
  void *result = wmempcpy(a, 0, n); // expected-warning{{Null pointer passed as 2nd argument to memory copy function}}
  clang_analyzer_eval(result == a); // no-warning (above is fatal)
}

void wmempcpy_unknownable_size (wchar_t *src, float n) {
  wchar_t a[4];
  // This used to crash because we don't model floats.
  wmempcpy(a, src, (size_t)n);
}

//===----------------------------------------------------------------------===
// wmemmove()
//===----------------------------------------------------------------------===

#define wmemmove BUILTIN(wmemmove)
wchar_t *wmemmove(wchar_t *s1, const wchar_t *s2, size_t n);

void wmemmove0 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[4] = {0};

  wmemmove(dst, src, 4); // no-warning

  clang_analyzer_eval(wmemmove(dst, src, 4) == dst); // expected-warning{{TRUE}}

  clang_analyzer_eval(dst[0] != 0); // expected-warning{{UNKNOWN}}
}

void wmemmove1 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[10];

  wmemmove(dst, src, 5); // expected-warning{{out-of-bound}}
}

void wmemmove2 (void) {
  wchar_t src[] = {1, 2, 3, 4};
  wchar_t dst[1];

  wmemmove(dst, src, 4); // expected-warning{{Memory copy function overflows the destination buffer}}
}

//===----------------------------------------------------------------------===
// wmemcmp()
//===----------------------------------------------------------------------===

#define wmemcmp BUILTIN(wmemcmp)
int wmemcmp(const wchar_t *s1, const wchar_t *s2, size_t n);

void wmemcmp0 (void) {
  wchar_t a[] = {1, 2, 3, 4};
  wchar_t b[4] = { 0 };

  wmemcmp(a, b, 4); // no-warning
}

void wmemcmp1 (void) {
  wchar_t a[] = {1, 2, 3, 4};
  wchar_t b[10] = { 0 };

  wmemcmp(a, b, 5); // expected-warning{{out-of-bound}}
}

void wmemcmp2 (void) {
  wchar_t a[] = {1, 2, 3, 4};
  wchar_t b[1] = { 0 };

  wmemcmp(a, b, 4); // expected-warning{{out-of-bound}}
}

void wmemcmp3 (void) {
  wchar_t a[] = {1, 2, 3, 4};

  clang_analyzer_eval(wmemcmp(a, a, 4) == 0); // expected-warning{{TRUE}}
}

void wmemcmp4 (wchar_t *input) {
  wchar_t a[] = {1, 2, 3, 4};

  clang_analyzer_eval(wmemcmp(a, input, 4) == 0); // expected-warning{{UNKNOWN}}
}

void wmemcmp5 (wchar_t *input) {
  wchar_t a[] = {1, 2, 3, 4};

  clang_analyzer_eval(wmemcmp(a, 0, 0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wmemcmp(0, a, 0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wmemcmp(a, input, 0) == 0); // expected-warning{{TRUE}}
}

void wmemcmp6 (wchar_t *a, wchar_t *b, size_t n) {
  int result = wmemcmp(a, b, n);
  if (result != 0)
    clang_analyzer_eval(n != 0); // expected-warning{{TRUE}}
  // else
  //   analyzer_assert_unknown(n == 0);

  // We can't do the above comparison because n has already been constrained.
  // On one path n == 0, on the other n != 0.
}

int wmemcmp7 (wchar_t *a, size_t x, size_t y, size_t n) {
  // We used to crash when either of the arguments was unknown.
  return wmemcmp(a, &a[x*y], n) +
         wmemcmp(&a[x*y], a, n);
}

int wmemcmp8(wchar_t *a, size_t n) {
  wchar_t *b = 0;
  // Do not warn about the first argument!
  return wmemcmp(a, b, n); // expected-warning{{Null pointer passed as 2nd argument to memory comparison function}}
}

//===----------------------------------------------------------------------===
// wcslen()
//===----------------------------------------------------------------------===

#define wcslen BUILTIN(wcslen)
size_t wcslen(const wchar_t *s);

void wcslen_constant0(void) {
  clang_analyzer_eval(wcslen(L"123") == 3); // expected-warning{{TRUE}}
}

void wcslen_constant1(void) {
  const wchar_t *a = L"123";
  clang_analyzer_eval(wcslen(a) == 3); // expected-warning{{TRUE}}
}

void wcslen_constant2(wchar_t x) {
  wchar_t a[] = L"123";
  clang_analyzer_eval(wcslen(a) == 3); // expected-warning{{TRUE}}

  a[0] = x;
  clang_analyzer_eval(wcslen(a) == 3); // expected-warning{{UNKNOWN}}
}

size_t wcslen_null(void) {
  return wcslen(0); // expected-warning{{Null pointer passed as 1st argument to string length function}}
}

size_t wcslen_fn(void) {
  return wcslen((wchar_t*)&wcslen_fn); // expected-warning{{Argument to string length function is the address of the function 'wcslen_fn', which is not a null-terminated string}}
}

size_t wcslen_nonloc(void) {
label:
  return wcslen((wchar_t*)&&label); // expected-warning{{Argument to string length function is the address of the label 'label', which is not a null-terminated string}}
}

void wcslen_subregion(void) {
  struct two_strings { wchar_t a[2], b[2]; };
  extern void use_two_strings(struct two_strings *);

  struct two_strings z;
  use_two_strings(&z);

  size_t a = wcslen(z.a);
  z.b[0] = 5;
  size_t b = wcslen(z.a);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  use_two_strings(&z);

  size_t c = wcslen(z.a);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

extern void use_string(wchar_t *);
void wcslen_argument(wchar_t *x) {
  size_t a = wcslen(x);
  size_t b = wcslen(x);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  use_string(x);

  size_t c = wcslen(x);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

extern wchar_t global_str[];
void wcslen_global(void) {
  size_t a = wcslen(global_str);
  size_t b = wcslen(global_str);
  if (a == 0) {
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
    // Make sure clang_analyzer_eval does not invalidate globals.
    clang_analyzer_eval(wcslen(global_str) == 0); // expected-warning{{TRUE}}
  }

  // Call a function with unknown effects, which should invalidate globals.
  use_string(0);

  size_t c = wcslen(global_str);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void wcslen_indirect(wchar_t *x) {
  size_t a = wcslen(x);
  wchar_t *p = x;
  wchar_t **p2 = &p;
  size_t b = wcslen(x);
  if (a == 0)
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}

  extern void use_string_ptr(wchar_t*const*);
  use_string_ptr(p2);

  size_t c = wcslen(x);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void wcslen_indirect2(wchar_t *x) {
  size_t a = wcslen(x);
  wchar_t *p = x;
  wchar_t **p2 = &p;
  extern void use_string_ptr2(wchar_t**);
  use_string_ptr2(p2);

  size_t c = wcslen(x);
  if (a == 0)
    clang_analyzer_eval(c == 0); // expected-warning{{UNKNOWN}}
}

void wcslen_liveness(const wchar_t *x) {
  if (wcslen(x) < 5)
    return;
  clang_analyzer_eval(wcslen(x) < 5); // expected-warning{{FALSE}}
}


size_t wcslenWrapper(const wchar_t *str) {
  return wcslen(str);
}

extern void invalidate(wchar_t *s);

void testwcslenCallee(void) {
  wchar_t str[42];
  invalidate(str);
  size_t lenBefore = wcslenWrapper(str);
  invalidate(str);
  size_t lenAfter = wcslenWrapper(str);
  clang_analyzer_eval(lenBefore == lenAfter); // expected-warning{{UNKNOWN}}
}

//===----------------------------------------------------------------------===
// wcsnlen()
//===----------------------------------------------------------------------===

size_t wcsnlen(const wchar_t *s, size_t maxlen);

void wcsnlen_constant0(void) {
  clang_analyzer_eval(wcsnlen(L"123", 10) == 3); // expected-warning{{TRUE}}
}

void wcsnlen_constant1(void) {
  const wchar_t *a = L"123";
  clang_analyzer_eval(wcsnlen(a, 10) == 3); // expected-warning{{TRUE}}
}

void wcsnlen_constant2(char x) {
  wchar_t a[] = L"123";
  clang_analyzer_eval(wcsnlen(a, 10) == 3); // expected-warning{{TRUE}}
  a[0] = x;
  clang_analyzer_eval(wcsnlen(a, 10) == 3); // expected-warning{{UNKNOWN}}
}

void wcsnlen_constant4(void) {
  clang_analyzer_eval(wcsnlen(L"123456", 3) == 3); // expected-warning{{TRUE}}
}

void wcsnlen_constant5(void) {
  const wchar_t *a = L"123456";
  clang_analyzer_eval(wcsnlen(a, 3) == 3); // expected-warning{{TRUE}}
}

void wcsnlen_constant6(char x) {
  wchar_t a[] = L"123456";
  clang_analyzer_eval(wcsnlen(a, 3) == 3); // expected-warning{{TRUE}}
  a[0] = x;
  clang_analyzer_eval(wcsnlen(a, 3) == 3); // expected-warning{{UNKNOWN}}
}

size_t wcsnlen_null(void) {
  return wcsnlen(0, 3); // expected-warning{{Null pointer passed as 1st argument to string length function}}
}

size_t wcsnlen_fn(void) {
  return wcsnlen((wchar_t*)&wcsnlen_fn, 3); // expected-warning{{Argument to string length function is the address of the function 'wcsnlen_fn', which is not a null-terminated string}}
}

size_t wcsnlen_nonloc(void) {
label:
  return wcsnlen((wchar_t*)&&label, 3); // expected-warning{{Argument to string length function is the address of the label 'label', which is not a null-terminated string}}
}

void wcsnlen_zero(void) {
  clang_analyzer_eval(wcsnlen(L"abc", 0) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcsnlen(0, 0) == 0); // expected-warning{{TRUE}}
}

size_t wcsnlen_compound_literal(void) {
  // This used to crash because we don't model the string lengths of
  // compound literals.
  return wcsnlen((wchar_t[]) { 'a', 'b', 0 }, 1);
}

size_t wcsnlen_unknown_limit(float f) {
  // This used to crash because we don't model the integer values of floats.
  return wcsnlen(L"abc", (int)f);
}

void wcsnlen_is_not_wcslen(wchar_t *x) {
  clang_analyzer_eval(wcsnlen(x, 10) == wcslen(x)); // expected-warning{{UNKNOWN}}
}

void wcsnlen_at_limit(wchar_t *x) {
  size_t len = wcsnlen(x, 10);
  clang_analyzer_eval(len <= 10); // expected-warning{{TRUE}}
  clang_analyzer_eval(len == 10); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(len < 10); // expected-warning{{UNKNOWN}}
}

void wcsnlen_at_actual(size_t limit) {
  size_t len = wcsnlen(L"abc", limit);
  clang_analyzer_eval(len <= 3); // expected-warning{{TRUE}}
  // This is due to eager assertion in wcsnlen.
  if (limit == 0) {
    clang_analyzer_eval(len == 0); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(len == 3); // expected-warning{{UNKNOWN}}
    clang_analyzer_eval(len < 3); // expected-warning{{UNKNOWN}}
  }
}

//===----------------------------------------------------------------------===
// other tests
//===----------------------------------------------------------------------===

static const wchar_t w_str[] = L"Hello world";

void wmemcpy_sizeof(void) {
  wchar_t a[32];
  wmemcpy(a, w_str, sizeof(w_str) / sizeof(w_str[0]));
  wmemcpy(a, w_str, (sizeof(w_str) / sizeof(w_str[0])) + 1); // expected-warning {{Memory copy function accesses out-of-bound array element}}
}

void wmemcpy_wcslen(void) {
  wchar_t a[32];
  // FIXME: This should work with 'w_str' instead of 'w_str1'
  const wchar_t w_str1[] = L"Hello world";
  wmemcpy(a, w_str1, wcslen(w_str1) + 1);
  wmemcpy(a, w_str1, wcslen(w_str1) + 2); // expected-warning {{Memory copy function accesses out-of-bound array element}}
}
