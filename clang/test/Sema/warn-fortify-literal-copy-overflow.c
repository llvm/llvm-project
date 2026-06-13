// RUN: %clang_cc1 -fsyntax-only -Wfortify-source -verify %s

char *strcpy(char *, const char *);

void literal_strcpy_overflow(void) {
  char buf[4];
  char ok[5];
  strcpy(buf, "abcd"); // expected-warning{{copying 5 bytes into buffer of size 4 (including null terminator)}}
  strcpy(ok, "abcd");
}
