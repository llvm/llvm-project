// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring -verify %s

// expected-no-diagnostics

typedef __SIZE_TYPE__ size_t;

extern char* strnlen(const char*, size_t);
char** q;

void f(const char *s)
{
  extern char a[8];
  q[0] = strnlen(a, 7); // no crash
}
