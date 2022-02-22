// RUN: %clang_analyze_cc1 -w -analyzer-checker=unix.cstring.BadSizeArg \
// RUN:                    -verify %s

// expected-no-diagnostics

typedef __SIZE_TYPE__ size_t;
// The last parameter is normally size_t but the test is about the abnormal
// situation when it's not a size_t.
size_t strlcpy(char *, const char *, void (*)(void));

void foo(void);

void testWeirdDecls(const char *src) {
  char dst[10];
  strlcpy(dst, src, foo); // no-crash
  strlcpy(dst, src, &foo); // no-crash
}
