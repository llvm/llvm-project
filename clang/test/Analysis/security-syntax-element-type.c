// RUN: %clang_analyze_cc1 -fno-builtin -Wno-error=incompatible-pointer-types -analyzer-checker=security.insecureAPI.strcpy -verify %s

typedef __WCHAR_TYPE__ test_wchar_t;

char *strcpy(char *dst, const char *src); // expected-note{{passing argument to parameter 'src' here}}
test_wchar_t *wcscpy(test_wchar_t *dst, const test_wchar_t *src); // expected-note{{passing argument to parameter 'src' here}}

// Mismatched type: narrow destination and wide-character literal.
void test_strcpy_wide_literal(void) {
  char dst[100];
  strcpy(dst, L"hello"); // expected-warning{{incompatible pointer types}} expected-warning{{Call to function 'strcpy' is insecure}}
}

// Mismatched type: wide destination and narrow-character literal.
void test_wcscpy_narrow_literal(void) {
  test_wchar_t dst[100];
  wcscpy(dst, "hello"); // expected-warning{{incompatible pointer types}} expected-warning{{Call to function 'wcscpy' is insecure}}
}

// Matching types: the destination has enough space.
void test_strcpy_matching_safe(void) {
  char dst[10];
  strcpy(dst, "hello");
}

// Matching types: the destination is too small.
void test_strcpy_matching_unsafe(void) {
  char dst[4];
  strcpy(dst, "hello"); // expected-warning{{Call to function 'strcpy' is insecure}}
}
