
// RUN: %clang_analyze_cc1 -fno-builtin -analyzer-checker=security.insecureAPI.strcpy -verify %s

// expected-no-diagnostics

typedef __WCHAR_TYPE__ test_wchar_t;

// Wide-character function declared with narrow-character parameters.
char *wcscpy(char *dst, const char *src);

void test_wcscpy_narrow_declaration(char *src) {
  char dst[10];
  wcscpy(dst, src);
}

// Narrow-character function declared with wide-character parameters.
test_wchar_t *strcpy(test_wchar_t *dst, const test_wchar_t *src);

void test_strcpy_wide_declaration(test_wchar_t *src) {
  test_wchar_t dst[10];
  strcpy(dst, src);
}

// Wide-character Windows function declared with narrow-character parameters.
char *lstrcpyW(char *dst, const char *src);

void test_lstrcpyW_narrow_declaration(char *src) {
  char dst[10];
  lstrcpyW(dst, src);
}

// Narrow-character Windows function declared with wide-character parameters.
test_wchar_t *lstrcatA(test_wchar_t *dst, const test_wchar_t *src);

void test_lstrcatA_wide_declaration(test_wchar_t *src) {
  test_wchar_t dst[10];
  lstrcatA(dst, src);
}
