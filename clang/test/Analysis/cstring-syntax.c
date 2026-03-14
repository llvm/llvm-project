// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.BadSizeArg -verify %s\
// RUN:                    -Wno-strncat-size -Wno-sizeof-pointer-memaccess     \
// RUN:                    -Wno-strlcpy-strlcat-size -Wno-sizeof-array-argument
// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.BadSizeArg -verify %s\
// RUN:                    -Wno-strncat-size -Wno-sizeof-pointer-memaccess     \
// RUN:                    -Wno-strlcpy-strlcat-size -Wno-sizeof-array-argument\
// RUN:                    -triple armv7-a15-linux
// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.BadSizeArg -verify %s\
// RUN:                    -Wno-strncat-size -Wno-sizeof-pointer-memaccess     \
// RUN:                    -Wno-strlcpy-strlcat-size -Wno-sizeof-array-argument\
// RUN:                    -triple aarch64_be-none-linux-gnu
// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.BadSizeArg -verify %s\
// RUN:                    -Wno-strncat-size -Wno-sizeof-pointer-memaccess     \
// RUN:                    -Wno-strlcpy-strlcat-size -Wno-sizeof-array-argument\
// RUN:                    -triple i386-apple-darwin10

typedef __SIZE_TYPE__ size_t;
char  *strncat(char *, const char *, size_t);
size_t strlen (const char *s);
size_t strlcpy(char *, const char *, size_t);
size_t strlcat(char *, const char *, size_t);

void testStrncat(const char *src) {
  char dest[10];
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - 1); // expected-warning {{Potential buffer overflow. Replace with 'sizeof(dest) - strlen(dest) - 1' or use a safer 'strlcat' API}}
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest)); // expected-warning {{Potential buffer overflow. Replace with}}
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - strlen(dest)); // expected-warning {{Potential buffer overflow. Replace with}}
  strncat(dest, src, sizeof(src)); // expected-warning {{Potential buffer overflow. Replace with}}
  // Should not crash when sizeof has a type argument.
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(char));
}

void testStrlcpy(const char *src) {
  char dest[10];
  size_t destlen = sizeof(dest);
  size_t srclen = sizeof(src);
  size_t badlen = 20;
  size_t ulen;
  strlcpy(dest, src, sizeof(dest));
  strlcpy(dest, src, destlen);
  strlcpy(dest, src, 10);
  strlcpy(dest, src, 20); // expected-warning {{The third argument allows to potentially copy more bytes than it should. Replace with the value sizeof(dest) or lower}}
  strlcpy(dest, src, badlen); // expected-warning {{The third argument allows to potentially copy more bytes than it should. Replace with the value sizeof(dest) or lower}}
  strlcpy(dest, src, ulen);
  strlcpy(dest + 5, src, 5);
  strlcpy(dest + 5, src, 10); // expected-warning {{The third argument allows to potentially copy more bytes than it should. Replace with the value sizeof(<destination buffer>) or lower}}
  strlcpy(dest, "aaaaaaaaaaaaaaa", 10); // no-warning
}

void testStrlcat(const char *src) {
  char dest[10];
  size_t badlen = 20;
  size_t ulen;
  strlcpy(dest, "aaaaa", sizeof("aaaaa") - 1);
  strlcat(dest, "bbbb", (sizeof("bbbb") - 1) - sizeof(dest) - 1);
  strlcpy(dest, "012345678", sizeof(dest));
  strlcat(dest, "910", sizeof(dest));
  strlcpy(dest, "0123456789", sizeof(dest));
  strlcpy(dest, "0123456789", sizeof(dest));
  strlcat(dest, "0123456789", badlen / 2);
  strlcat(dest, "0123456789", badlen); // expected-warning {{The third argument allows to potentially copy more bytes than it should. Replace with the value sizeof(dest) or lower}}
  strlcat(dest, "0123456789", badlen - strlen(dest) - 1);
  strlcat(dest, src, ulen);
  strlcpy(dest, src, 5);
  strlcat(dest + 5, src, badlen); // expected-warning {{The third argument allows to potentially copy more bytes than it should. Replace with the value sizeof(<destination buffer>) or lower}}
  strlcat(dest, "aaaaaaaaaaaaaaa", 10); // no-warning
}
