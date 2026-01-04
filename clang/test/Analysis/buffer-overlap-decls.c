// RUN: %clang_analyze_cc1 -verify %s -Wno-incompatible-library-redeclaration \
// RUN:   -analyzer-checker=alpha.unix.cstring.BufferOverlap
// expected-no-diagnostics

typedef typeof(sizeof(int)) size_t;

void memcpy(int dst, int src, size_t size);

void test_memcpy_proxy() {
  memcpy(42, 42, 42); // no-crash
}

void strcpy(int dst, char *src);

void test_strcpy_proxy() {
  strcpy(42, (char *)42); // no-crash
}

void strxfrm(int dst, char *src, size_t size);

void test_strxfrm_proxy() {
  strxfrm(42, (char *)42, 42); // no-crash
}
