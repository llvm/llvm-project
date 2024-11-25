// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-disable-checker=alpha.unix.cstring.UninitializedRead \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// RUN: %clang_analyze_cc1 -verify %s -DUSE_BUILTINS \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-disable-checker=alpha.unix.cstring.UninitializedRead \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false
//
// Enabling the malloc checker enables some of the buffer-checking portions
// of the C-string checker.

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

#define NULL (0)

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

void clang_analyzer_eval(int);

void escape(wchar_t*);

void *malloc(size_t);
int scanf(const char *restrict format, ...);
void free(void *);

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
// memcmp()
// Checking here, that the non-wide-char version still works as expected.
//===----------------------------------------------------------------------===
int memcmp(const void *s1, const void *s2, size_t n);

int memcmp_same_buffer_not_oob (void) {
  char a[] = {1, 2, 3, 4};
  return memcmp(a, a, 4); // no-warning
}

int memcmp_same_buffer_oob (void) {
  char a[] = {1, 2, 3, 4};
  return memcmp(a, a, 5); // expected-warning{{Memory comparison function accesses out-of-bound array element}}
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

int wmemcmp_same_buffer_not_oob (void) {
  wchar_t a[] = {1, 2, 3, 4};
  return wmemcmp(a, a, 4); // no-warning
}

int wmemcmp_same_buffer_oob (void) {
  wchar_t a[] = {1, 2, 3, 4};
  return wmemcmp(a, a, 5); // expected-warning{{Memory comparison function accesses out-of-bound array element}}
}

void wmemcmp_same_buffer_value (void) {
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
  else {
    // result can be 0 regardless of the value of n.
    // However, in the model of wmemcmp, analyzer splits state on n being 0 and not.
    // For that reason we get two results TRUE and FALSE instead of one UNKNOWN.
    clang_analyzer_eval(n == 0);
    // expected-warning@-1{{TRUE}}
    // expected-warning@-2{{FALSE}}
  }
}

int wmemcmp7 (wchar_t *a, size_t x, size_t y, size_t n) {
  // We used to crash when either of the arguments was unknown.
  return wmemcmp(a, &a[x*y], n) +
         wmemcmp(&a[x*y], a, n);
}

int wmemcmp8(wchar_t *a, size_t n) {
  wchar_t *b = 0;
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

//===----------------------------------------------------------------------===
// wcscpy()
//===----------------------------------------------------------------------===

wchar_t* wcscpy(wchar_t *restrict s1, const wchar_t *restrict s2);

void wcscpy_null_dst(wchar_t *x) {
  wcscpy(NULL, x); // expected-warning{{Null pointer passed as 1st argument to string copy function}}
}

void wcscpy_null_src(wchar_t *x) {
  wcscpy(x, NULL); // expected-warning{{Null pointer passed as 2nd argument to string copy function}}
}

void wcscpy_fn(wchar_t *x) {
  wcscpy(x, (wchar_t*)&wcscpy_fn); // expected-warning{{Argument to string copy function is the address of the function 'wcscpy_fn', which is not a null-terminated string}}
}

void wcscpy_fn_const(wchar_t *x) {
  wcscpy(x, (const wchar_t*)&wcscpy_fn); // expected-warning{{Argument to string copy function is the address of the function 'wcscpy_fn', which is not a null-terminated string}}
}

void wcscpy_label(wchar_t *x) {
label:
  wcscpy(x, (const wchar_t*)&&label); // expected-warning{{Argument to string copy function is the address of the label 'label', which is not a null-terminated string}}
}

extern int globalInt;
void wcscpy_effects(wchar_t *x, wchar_t *y) {
  wchar_t x0 = x[0];
  globalInt = 42;

  clang_analyzer_eval(wcscpy(x, y) == x); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen(x) == wcslen(y)); // expected-warning{{TRUE}}
  clang_analyzer_eval(x0 == x[0]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(globalInt == 42); // expected-warning{{TRUE}}
}

void wcscpy_model_after_call() {
  wchar_t src[] = L"AAA";
  wchar_t dst[10];

  clang_analyzer_eval(wcslen(src) == 3); // expected-warning{{TRUE}}
  wcscpy(dst, src);
  clang_analyzer_eval(wcslen(dst) == 3); // expected-warning{{TRUE}}
}

void wcscpy_overflow(wchar_t *y) {
  wchar_t x[4];
  if (wcslen(y) == 4)
    wcscpy(x, y); // expected-warning{{String copy function overflows the destination buffer}}
}

void wcscpy_no_overflow(wchar_t *y) {
  wchar_t x[4];
  if (wcslen(y) == 3)
    wcscpy(x, y); // no-warning
}

void wcscpy_overlapping_local_arr(void) {
  wchar_t arr[10];
  escape(arr);
  if (wcslen(arr) != 5 || wcslen(arr + 1) != 4)
    return;
  // Given that arr points to a non-empty string,
  // arr and arr + 1 overlap, but we don't detect it.
  wcscpy(arr, arr + 1); // no-warning false negative
  // We can detect the exact match, however.
  wcscpy(arr, arr); // expected-warning{{overlapping}}
}

void wcscpy_overlapping_param(wchar_t* buf) {
  if (10 < wcslen(buf)) {
    wcscpy(buf, buf + 6); // no-warning false negative
  }
  wcscpy(buf, buf); // expected-warning{{overlapping}}
}

//===----------------------------------------------------------------------===
// wcsncpy()
//===----------------------------------------------------------------------===

wchar_t* wcsncpy(wchar_t *restrict s1, const wchar_t *restrict s2, size_t n);

void wcsncpy_null_dst(wchar_t *x) {
  wcsncpy(NULL, x, 5); // expected-warning{{Null pointer passed as 1st argument to string copy function}}
}

void wcsncpy_null_src(wchar_t *x) {
  wcsncpy(x, NULL, 5); // expected-warning{{Null pointer passed as 2nd argument to string copy function}}
}

void wcsncpy_fn(wchar_t *x) {
  wcsncpy(x, (wchar_t*)&wcsncpy_fn, 5); // expected-warning{{Argument to string copy function is the address of the function 'wcsncpy_fn', which is not a null-terminated string}}
}

void wcsncpy_effects(wchar_t *x, wchar_t *y) {
  wchar_t x0 = x[0];

  clang_analyzer_eval(wcsncpy(x, y, 5) == x); // expected-warning{{TRUE}}
  wcsncpy(x, y, 5);
  clang_analyzer_eval(wcslen(x) == wcslen(y)); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(x0 == x[0]); // expected-warning{{UNKNOWN}}
}

// Make sure the checker does not crash when the length argument is way beyond the
// extents of the source and dest arguments
void wcsncpy_cstringchecker_bounds_nocrash(void) {
  wchar_t *p = malloc(2 * sizeof(wchar_t));
  // sizeof(L"AAA") returns 4*sizeof(wchar_t), e.g., 16, which is longer than
  // the number of characters in L"AAA" - 4:
  wcsncpy(p, L"AAA", sizeof(L"AAA")); // expected-warning {{String copy function overflows the destination buffer}}
  free(p);
}

void wcsncpy_overflow(wchar_t *y) {
  wchar_t x[4];
  if (wcslen(y) == 4)
    wcsncpy(x, y, 5); // expected-warning {{String copy function overflows the destination buffer}}
}

void wcsncpy_overflow_from_sizearg_1() {
  wchar_t dst[3];
  wchar_t src[] = L"1";
  wcsncpy(dst, src, 5); // expected-warning {{String copy function overflows the destination buffer}}
}

void wcsncpy_overflow_from_sizearg_2(wchar_t *y) {
  wchar_t x[4];
  // From man page:
  // If the length wcslen(src) is smaller than n, the remaining wide characters
  // in the array pointed to by dest are filled with null wide characters.
  //
  // So, Exactly 5 wchars will be written even if wcslen is 3.
  // Hence, the following overflows even though y could fit into x.
  if (wcslen(y) == 3)
    wcsncpy(x, y, 5); // expected-warning {{String copy function overflows the destination buffer}}
}

void wcsncpy_overflow_from_src_1() {
  wchar_t dst[3];
  wchar_t src[] = L"1234";
  wcsncpy(dst, src, 5); // expected-warning {{String copy function overflows the destination buffer}}
}

void wcsncpy_overflow_from_src_2() {
  wchar_t dst[3];
  wchar_t src[] = L"123456";
  wcsncpy(dst, src, 5); // expected-warning {{String copy function overflows the destination buffer}}
}

void wcsncpy_no_overflow_no_null_term(wchar_t *y) {
  wchar_t x[4];
  if (wcslen(y) == 10)
    wcsncpy(x, y, 4); // no-warning
}

void wcsncpy_no_overflow_false_negative(wchar_t *y, int n) {
  if (n <= 4)
    return;

  // This generates no warning because
  // the built-in range-based solver has weak support for multiplication.
  // In particular it cannot see that
  //    { "symbol": "((reg_$0<int n>) - 1) * 4U", "range": "{ [0, 3] }" }
  //    { "symbol": "reg_$0<int n>", "range": "{ [41, 2147483647] }" }
  // constraints are incompatible
  wchar_t x[4];
  if (wcslen(y) == 3)
    wcsncpy(x, y, n); // no-warning - false negative
}

void wcsncpy_truncate(wchar_t *y) {
  wchar_t x[4];
  if (wcslen(y) == 4)
    wcsncpy(x, y, 3); // no-warning
}

void wcsncpy_no_truncate(wchar_t *y) {
  wchar_t x[4];
  if (wcslen(y) == 3)
    wcsncpy(x, y, 3); // no-warning
}

void wcsncpy_exactly_matching_buffer(wchar_t *y) {
  wchar_t x[4];
  wcsncpy(x, y, 4); // no-warning

  // wcsncpy does not null-terminate, so we have no idea what the strlen is
  // after this.
  clang_analyzer_eval(wcslen(x) > 4); // expected-warning{{UNKNOWN}}
}

void wcsncpy_zero(wchar_t *src) {
  wchar_t dst[] = L"123";
  wcsncpy(dst, src, 0); // no-warning
}

void wcsncpy_empty(void) {
  wchar_t dst[] = L"123";
  wchar_t src[] = L"";
  wcsncpy(dst, src, 4); // no-warning
}

void wcsncpy_overlapping_local_arr(int way) {
  wchar_t arr[10];
  escape(arr);
  switch(way) {
    case 0:
      wcsncpy(arr, arr + way, 5); // expected-warning{{overlapping}}
    case 1:
      wcsncpy(arr, arr + way, 5); // expected-warning{{overlapping}}
    case 4:
      wcsncpy(arr, arr + way, 5); // expected-warning{{overlapping}}
    case 5:
      wcsncpy(arr, arr + way, 5);
  }
}

void wcsncpy_overlapping_param1(wchar_t* buf) {
  wcsncpy(buf, buf + 6, 10); // expected-warning{{overlapping}}
}

void wcsncpy_overlapping_param2(wchar_t* buf1, wchar_t* buf2) {
  if (buf1 == buf2)
    wcsncpy(buf1, buf2, 10); // expected-warning{{overlapping}}

  // False negatives:
  if (buf1 + 6 == buf2)
    wcsncpy(buf1, buf2, 10); // no-warning
  if (buf2 - buf1 < 6)
    wcsncpy(buf1, buf2, 10); // no-warning
}

//===----------------------------------------------------------------------===
// wcscat()
//===----------------------------------------------------------------------===

wchar_t *wcscat(wchar_t *restrict s1, const wchar_t *restrict s2);

void wcscat_null_dst(wchar_t *x) {
  wcscat(NULL, x); // expected-warning{{Null pointer passed as 1st argument to string concatenation function}}
}

void wchar_t_null_src(wchar_t *x) {
  wcscat(x, NULL); // expected-warning{{Null pointer passed as 2nd argument to string concatenation function}}
}

void wcscat_fn(wchar_t *x) {
  wcscat(x, (wchar_t*)&wcscat_fn); // expected-warning{{Argument to string concatenation function is the address of the function 'wcscat_fn', which is not a null-terminated string}}
}

void wcscat_effects(wchar_t *y) {
  wchar_t x[8] = L"123";
  size_t orig_len = wcslen(x);
  wchar_t x0 = x[0];

  if (wcslen(y) != 4)
    return;

  clang_analyzer_eval(wcscat(x, y) == x); // expected-warning{{TRUE}}

  clang_analyzer_eval((int)wcslen(x) == (orig_len + wcslen(y))); // expected-warning{{TRUE}}
}

void wcscat_overflow_0(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 4)
    wcscat(x, y); // expected-warning{{String concatenation function overflows the destination buffer}}
}

void wcscat_overflow_1(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 3)
    wcscat(x, y); // expected-warning{{String concatenation function overflows the destination buffer}}
}

void wcscat_overflow_2(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 2)
    wcscat(x, y); // expected-warning{{String concatenation function overflows the destination buffer}}
}

void wcscat_no_overflow(wchar_t *y) {
  wchar_t x[5] = L"12";
  if (wcslen(y) == 2)
    wcscat(x, y); // no-warning
}

void wcscat_unknown_dst_length(wchar_t *dst) {
  wcscat(dst, L"1234");
  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}
}

void wcscat_unknown_src_length(wchar_t *src, int offset) {
  wchar_t dst[8] = L"1234";
  wcscat(dst, &src[offset]);
  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}
}

void wcscat_too_big(wchar_t *dst, wchar_t *src) {
  // We assume this can never actually happen, so we don't get a warning.
  if (wcslen(dst) != (((size_t)0) - 2))
    return;
  if (wcslen(src) != 2)
    return;
  wcscat(dst, src);
}

void wcscat_overlapping_local_arr(void) {
  wchar_t arr[10];
  escape(arr);
  if (wcslen(arr) != 5 || wcslen(arr + 1) != 4)
    return;
  // Given that arr points to a non-empty string,
  // arr and arr + 1 overlap, but we don't detect it.
  wcscat(arr, arr + 1); // no-warning false negative
  // We can detect the exact match, however.
  wcscat(arr, arr); // expected-warning{{overlapping}}
}

void wcscat_overlapping_param(wchar_t* buf) {
  if (10 < wcslen(buf)) {
    wcscat(buf, buf + 6); // no-warning false negative
  }
  wcscat(buf, buf); // expected-warning{{overlapping}}
}

//===----------------------------------------------------------------------===
// wcsncat()
//===----------------------------------------------------------------------===

wchar_t *wcsncat(wchar_t *restrict s1, const wchar_t *restrict s2, size_t n);

void wcsncat_null_dst(wchar_t *x) {
  wcsncat(NULL, x, 4); // expected-warning{{Null pointer passed as 1st argument to string concatenation function}}
}

void wcsncat_null_src(wchar_t *x) {
  wcsncat(x, NULL, 4); // expected-warning{{Null pointer passed as 2nd argument to string concatenation function}}
}

void wcsncat_fn(wchar_t *x) {
  wcsncat(x, (wchar_t*)&wcsncat_fn, 4); // expected-warning{{Argument to string concatenation function is the address of the function 'wcsncat_fn', which is not a null-terminated string}}
}

void wcsncat_effects(wchar_t *y) {
  wchar_t x[8] = L"123";
  size_t orig_len = wcslen(x);
  wchar_t x0 = x[0];

  if (wcslen(y) != 4)
    return;

  clang_analyzer_eval(wcsncat(x, y, wcslen(y)) == x); // expected-warning{{TRUE}}

  clang_analyzer_eval(wcslen(x) == (orig_len + wcslen(y))); // expected-warning{{TRUE}}
}

void wcsncat_overflow_0(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 4)
    wcsncat(x, y, wcslen(y)); // expected-warning {{String concatenation function overflows the destination buffer}}
}

void wcsncat_overflow_1(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 3)
    wcsncat(x, y, wcslen(y)); // expected-warning {{String concatenation function overflows the destination buffer}}
}

void wcsncat_overflow_2(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 2)
    wcsncat(x, y, wcslen(y)); // expected-warning {{String concatenation function overflows the destination buffer}}
}

void wcsncat_overflow_3(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 4)
    wcsncat(x, y, 2); // expected-warning {{String concatenation function overflows the destination buffer}}
}

void wcsncat_no_overflow_1(wchar_t *y) {
  wchar_t x[5] = L"12";
  if (wcslen(y) == 2)
    wcsncat(x, y, wcslen(y)); // no-warning
}

void wcsncat_no_overflow_2(wchar_t *y) {
  wchar_t x[4] = L"12";
  if (wcslen(y) == 4)
    wcsncat(x, y, 1); // no-warning
}

void wcsncat_no_overflow_fn(wchar_t *y, unsigned n) {
  if (n < 2) {
    return;
  }

  // The analyzer does not take advantage of the known fact that
  // length of x is 2 and length of y is 4 to conclude that it does
  // not fit into x that has only 4 elements.
  wchar_t x[4] = L"12";
  if (wcslen(y) == 4)
    wcsncat(x, y, n); // no-warning
}

void wcsncat_unknown_dst_length(wchar_t *dst) {
  wcsncat(dst, L"1234", 5);
  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}
}

void wcsncat_unknown_src_length(wchar_t *src) {
  wchar_t dst[8] = L"1234";
  wcsncat(dst, src, 3);
  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}
  // Limitation: No modeling for the upper bound.
  clang_analyzer_eval(wcslen(dst) <= 10); // expected-warning{{UNKNOWN}}

  wchar_t dst2[8] = L"1234";
  wcsncat(dst2, src, 4); // expected-warning {{String concatenation function overflows the destination buffer}}
}

void wcsncat_unknown_src_length_with_offset(wchar_t *src, int offset) {
  wchar_t dst[8] = L"1234";
  wcsncat(dst, &src[offset], 3);
  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}

  wchar_t dst2[8] = L"1234";
  wcsncat(dst2, &src[offset], 4); // expected-warning {{String concatenation function overflows the destination buffer}}
}

void wcsncat_unknown_limit(unsigned limit) {
  wchar_t dst[6] = L"1234";
  wchar_t src[] = L"567";
  wcsncat(dst, src, limit); // no-warning

  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen(dst) == 4); // expected-warning{{UNKNOWN}}
  // Limitation: No modeling for the upper bound
  clang_analyzer_eval(wcslen(dst) < 10); // expected-warning{{UNKNOWN}}
}

void wcsncat_unknown_float_limit(float limit) {
  wchar_t dst[6] = L"1234";
  wchar_t src[] = L"567";
  wcsncat(dst, src, (size_t)limit); // no-warning

  clang_analyzer_eval(wcslen(dst) >= 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen(dst) == 4); // expected-warning{{UNKNOWN}}
}

void wcsncat_too_big(wchar_t *dst, wchar_t *src) {
  // We assume this will never actually happen, so we don't get a warning.
  if (wcslen(dst) != (((size_t)0) - 2))
    return;
  if (wcslen(src) != 2)
    return;
  wcsncat(dst, src, 2);
}

void wcsncat_zero(wchar_t *src) {
  wchar_t dst[] = L"123";
  wcsncat(dst, src, 0); // no-warning
}

void wcsncat_zero_unknown_dst(wchar_t *dst, wchar_t *src) {
  wcsncat(dst, src, 0); // no-warning
}

void wcsncat_empty(void) {
  wchar_t dst[8] = L"123";
  wchar_t src[] = L"";
  wcsncat(dst, src, 4); // no-warning
}

void wcsncat_overlapping_local_arr(int way) {
  wchar_t arr[10];
  escape(arr);
  switch(way) {
    case 0:
      wcsncat(arr, arr + way, 5); // expected-warning{{overlapping}}
    case 1:
      wcsncat(arr, arr + way, 5); // expected-warning{{overlapping}}
    case 4:
      wcsncat(arr, arr + way, 5); // expected-warning{{overlapping}}
    case 5:
      wcsncat(arr, arr + way, 5);
  }
}

void wcsncat_overlapping_param1(wchar_t* buf) {
  wcsncat(buf, buf + 6, 10); // expected-warning{{overlapping}}
}

void wcsncat_overlapping_param2(wchar_t* buf1, wchar_t* buf2) {
  if (buf1 == buf2)
    wcsncat(buf1, buf2, 10); // expected-warning{{overlapping}}

  // False negatives:
  if (buf1 + 6 == buf2)
    wcsncat(buf1, buf2, 10); // no-warning
  if (buf2 - buf1 < 6)
    wcsncat(buf1, buf2, 10); // no-warning
}

//===----------------------------------------------------------------------===
// wcscmp()
//===----------------------------------------------------------------------===

#define wcscmp BUILTIN(wcscmp)
int wcscmp(const wchar_t  *string1, const wchar_t  *string2);

void wcscmp_check_modeling(void) {
  wchar_t x[] = L"aa";
  wchar_t y[] = L"a";
  clang_analyzer_eval(wcscmp(x, x) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcscmp(x, y) == 0); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(wcscmp(&x[1], x) == 0); // expected-warning{{UNKNOWN}}
}

void wcscmp_null_0(void) {
  wchar_t *x = NULL;
  wchar_t y[] = L"123";
  (void)wcscmp(x, y); // expected-warning{{Null pointer passed as 1st argument to string comparison function}}
}

void wcscmp_null_1(void) {
  wchar_t x[] = L"123";
  wchar_t *y = NULL;
  (void)wcscmp(x, y); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

union argument {
   char *f;
};

void function_pointer_cast_helper(wchar_t **a) {
  // Similar code used to crash for regular c-strings, see PR24951
  (void)wcscmp(L"Hi", *a);
}

void wcscmp_union_function_pointer_cast(union argument a) {
  void (*fPtr)(union argument *) = (void (*)(union argument *))function_pointer_cast_helper;

  fPtr(&a);
}

int wcscmp_null_argument(wchar_t *a) {
  wchar_t *b = 0;
  return wcscmp(a, b); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

//===----------------------------------------------------------------------===
// wcsncmp()
//===----------------------------------------------------------------------===

#define wcsncmp BUILTIN(wcsncmp)
int wcsncmp(const wchar_t  *string1, const wchar_t  *string2, size_t count);

void wcsncmp_check_modeling(void) {
  wchar_t x[] = L"aa";
  wchar_t y[] = L"a";
  clang_analyzer_eval(wcsncmp(x, x, 2) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcsncmp(x, y, 2) == 0); // expected-warning{{UNKNOWN}}
}

void wcsncmp_null_0(void) {
  wchar_t *x = NULL;
  wchar_t y[] = L"123";
  (void)wcsncmp(x, y, 3); // expected-warning{{Null pointer passed as 1st argument to string comparison function}}
}

void wcsncmp_null_1(void) {
  wchar_t x[] = L"123";
  wchar_t *y = NULL;
  (void)wcsncmp(x, y, 3); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

int wcsncmp_null_argument(wchar_t *a, size_t n) {
  wchar_t *b = 0;
  return wcsncmp(a, b, n); // expected-warning{{Null pointer passed as 2nd argument to string comparison function}}
}

//===----------------------------------------------------------------------===
// wmemset()
//===----------------------------------------------------------------------===

wchar_t *wmemset(wchar_t *s, wchar_t c, size_t n);

void wmemset1_char_array_null(void) {
  wchar_t str[] = L"abcd";
  clang_analyzer_eval(wcslen(str) == 4); // expected-warning{{TRUE}}
  wmemset(str, L'\0', 2);
  clang_analyzer_eval(wcslen(str) == 0); // expected-warning{{TRUE}}
}

void wmemset2_char_array_null(void) {
  wchar_t str[] = L"abcd";
  clang_analyzer_eval(wcslen(str) == 4); // expected-warning{{TRUE}}
  wmemset(str, L'\0', wcslen(str) + 1);
  clang_analyzer_eval(wcslen(str) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(str[2] == 0);      // expected-warning{{TRUE}}
}

void wmemset3_char_malloc_null(void) {
  wchar_t *str = (wchar_t *)malloc(10 * sizeof(wchar_t));
  wmemset(str + 1, '\0', 8);
  clang_analyzer_eval(str[1] == 0); // expected-warning{{UNKNOWN}}
  free(str);
}

void wmemset4_char_malloc_null(void) {
  wchar_t *str = (wchar_t *)malloc(10 * sizeof(wchar_t));
  //void *str = malloc(10 * sizeof(char));
  wmemset(str, '\0', 10);
  clang_analyzer_eval(str[1] == 0);      // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen(str) == 0); // expected-warning{{TRUE}}
  free(str);
}

void wmemset6_char_array_nonnull(void) {
  wchar_t str[] = L"abcd";
  clang_analyzer_eval(wcslen(str) == 4); // expected-warning{{TRUE}}
  wmemset(str, L'Z', 2);
  clang_analyzer_eval(str[0] == L'Z');   // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(wcslen(str) == 4); // expected-warning{{UNKNOWN}}
}

void wmemset_len_lowerbound(wchar_t *array) {
  clang_analyzer_eval(10 <= wcslen(array)); // expected-warning{{UNKNOWN}}
  wmemset(array, L'a', 10);
  clang_analyzer_eval(10 <= wcslen(array)); // expected-warning{{TRUE}}
}

void wmemset_zero_param(wchar_t *array) {
  wmemset(array, 0, 10);
  clang_analyzer_eval(wcslen(array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[0] == 0); // expected-warning{{UNKNOWN}}
}

void wmemset_zero_local() {
  wchar_t array[10];
  wmemset(array, 0, 10);
  clang_analyzer_eval(wcslen(array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[0] == 0); // expected-warning{{TRUE}}
}


struct POD_wmemset {
  int num;
  wchar_t c;
};

void wmemset10_struct(void) {
  struct POD_wmemset pod;
  wchar_t *str = (wchar_t *)&pod;
  pod.num = 1;
  pod.c = 1;
  clang_analyzer_eval(pod.num == 0); // expected-warning{{FALSE}}
  wmemset(str, 0, sizeof(struct POD_wmemset) / sizeof(wchar_t));
  clang_analyzer_eval(pod.num == 0); // expected-warning{{TRUE}}
}

void wmemset14_region_cast(void) {
  wchar_t *str = (wchar_t *)malloc(10 * sizeof(int));
  int *array = (int *)str;
  wmemset((wchar_t *)array, 0, 10 * sizeof(int) / sizeof(wchar_t));
  clang_analyzer_eval(str[10] == L'\0');              // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen((wchar_t *)array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen(str) == 0);              // expected-warning{{TRUE}}
  free(str);
}

void wmemset15_region_cast(void) {
  wchar_t *str = (wchar_t *)malloc(10 * sizeof(int));
  int *array = (int *)str;
  wmemset((wchar_t *)array, 0, 5 * sizeof(int) / sizeof(wchar_t));
  clang_analyzer_eval(str[10] == '\0');               // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(wcslen((wchar_t *)array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(wcslen(str) == 0);              // expected-warning{{TRUE}}
  free(str);
}

int wmemset20_scalar(void) {
  int *x = malloc(sizeof(int));
  *x = 10;
  wmemset((wchar_t *)x, 0, sizeof(int) / sizeof(wchar_t));
  int num = 1 / *x; // expected-warning{{Division by zero}}
  free(x);
  return num;
}

int wmemset21_scalar(void) {
  int *x = malloc(sizeof(int));
  wmemset((wchar_t *)x, 0, sizeof(int) / sizeof(wchar_t));
  int num = 1 / *x; // expected-warning{{Division by zero}}
  free(x);
  return num;
}

int wmemset211_long_scalar(void) {
  long long *x = malloc(sizeof(long long));
  wmemset((wchar_t *)x, 0, 1);
  // The memory region is wider than sizeof(wchar_t) * 1.
  // Limited modelling of wmemset simply invalidates the memory region
  // rather than writing it or part of it to 0.
  // So no division by zero or uninitialized read are reported.
  int num = 1 / *x; // no-warning
  free(x);
  return num;
}

void wmemset22_array(void) {
  int array[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  clang_analyzer_eval(array[1] == 2); // expected-warning{{TRUE}}
  wmemset((wchar_t *)array, 0, sizeof(array) / (sizeof(wchar_t)));
  clang_analyzer_eval(array[1] == 0); // expected-warning{{TRUE}}
}

void wmemset23_array_pod_object(void) {
  struct POD_wmemset array[10];
  array[1].num = 10;
  array[1].c = 'c';
  clang_analyzer_eval(array[1].num == 10); // expected-warning{{TRUE}}
  wmemset((wchar_t *)&array[1], 0, sizeof(struct POD_wmemset));
  clang_analyzer_eval(array[1].num == 0); // expected-warning{{UNKNOWN}}
}

void wmemset24_array_pod_object(void) {
  struct POD_wmemset array[10];
  array[1].num = 10;
  array[1].c = 'c';
  clang_analyzer_eval(array[1].num == 10); // expected-warning{{TRUE}}
  wmemset((wchar_t *)array, 0, sizeof(array) / (sizeof(wchar_t)));
  clang_analyzer_eval(array[1].num == 0); // expected-warning{{TRUE}}
}

void wmemset25_symbol(char c) {
  wchar_t array[10] = {1};
  if (c != 0)
    return;

  wmemset(array, c, 10);

  clang_analyzer_eval(wcslen(array) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(array[4] == 0); // expected-warning{{TRUE}}
}

void wmemset26_upper_UCHAR_MAX(void) {
  wchar_t array[10] = {1};

  // If cast to unsigned char, 0x400 would give 0
  // This test ensures that it does not happen
  wmemset(array, 0x400, 10);

  clang_analyzer_eval(wcslen(array) == 0);  // expected-warning{{FALSE}}
  clang_analyzer_eval(10 <= wcslen(array)); // expected-warning{{TRUE}}
  // The actual value is 0x400, but any non-0 is not modeled
  clang_analyzer_eval(array[4] == 0);       // expected-warning{{UNKNOWN}}
}

void wmemset_pseudo_zero_val_param(wchar_t *array) {
  wmemset(array, 0xff00, 10); // if cast to char, 0xff00 is '\0'
  clang_analyzer_eval(wcslen(array) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(array[0] == 0); // expected-warning{{UNKNOWN}}
}

void wmemset_pseudo_zero_val_local() {
  wchar_t array[10];
  wmemset(array, 0xff00, 10); // if cast to char, 0xff00 is '\0'
  clang_analyzer_eval(wcslen(array) == 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(array[0] == 0); // expected-warning{{UNKNOWN}}
}

void wmemset_almost_overflows() {
  wchar_t array[10] = {1};

  wmemset(array, 0, 10); // no-warning
}

void wmemset_overflows() {
  wchar_t array[10] = {1};

  wmemset(array, 0, 11); // expected-warning{{Memory set function overflows the destination buffer}}
}

//===----------------------------------------------------------------------===
// swprintf()
//===----------------------------------------------------------------------===

int swprintf(wchar_t* restrict ws, size_t n, const wchar_t* restrict format, ...);

void swprintf_null_dst(void) {
  swprintf(NULL, 10, L"%s", L"Hello"); // expected-warning{{Null pointer passed as 1st argument to 'swprintf'}}
}

void swprintf_null_dst_zero_size(void) {
  swprintf(NULL, 0, L"%s", L"Hello"); // no-warning: 0 size means no write will be done
}

void swprintf_null_dst_unknown_size(int size) {
  // If size is not known to be non-0, no check for the destination
  swprintf(NULL, size, L"%s", L"Hello"); // no-warning
  if (size == 0) {
    // If size is known to be 0, no check for dest buffer, no write access here
    swprintf(NULL, size, L"%s", L"Hello"); // no-warning
  } else {
    // If size is known to be non-0 it will likely try to write,
    // so warn on a null dest buffer.
    // This is the same code as in the beginning of the function where
    // it produced no report because there was no constraint on size.
    swprintf(NULL, size, L"%s", L"Hello"); // expected-warning{{Null pointer passed as 1st argument to 'swprintf'}}
  }
}

void swprintf_overlapping_param(wchar_t* buf) {
  swprintf(buf, 10, L"%s", buf); // no-warning false negative
}

void swprintf_overlapping_local_buf(void) {
  wchar_t buf[10];
  escape(buf);
  swprintf(buf, 10, L"%s", buf); // no-warning false negative
}
