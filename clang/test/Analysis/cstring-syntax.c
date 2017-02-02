// RUN: %clang_cc1 -analyze -analyzer-checker=unix.cstring.BadSizeArg -analyzer-store=region -Wno-strncat-size -Wno-strlcpy-strlcat-size -Wno-sizeof-array-argument -Wno-sizeof-pointer-memaccess -verify %s

typedef __SIZE_TYPE__ size_t;
char  *strncat(char *, const char *, size_t);
size_t strlen (const char *s);

void testStrncat(const char *src) {
  char dest[10];
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - 1); // expected-warning {{Potential buffer overflow. Replace with 'sizeof(dest) - strlen(dest) - 1' or use a safer 'strlcat' API}}
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest)); // expected-warning {{Potential buffer overflow. Replace with}}
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - strlen(dest)); // expected-warning {{Potential buffer overflow. Replace with}}
  strncat(dest, src, sizeof(src)); // expected-warning {{Potential buffer overflow. Replace with}}
  // Should not crash when sizeof has a type argument.
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(char));
}
