// RUN: %clang -std=c17 %s -o %t && %run %t
/// Test __isoc23_* for glibc 2.38+.
// RUN: %clang -std=c2x %s -o %t && %run %t

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

#define TESTL(func)                                                            \
  {                                                                            \
    char *end;                                                                 \
    long l = (long)func("42", &end, 0);                                        \
    assert(l == 42);                                                           \
    assert(*end == '\0');                                                      \
  }

#define TESTF(func)                                                            \
  {                                                                            \
    char *end;                                                                 \
    long l = (long)func("42", &end);                                           \
    assert(l == 42);                                                           \
    assert(*end == '\0');                                                      \
  }

#define WTESTL(func)                                                           \
  {                                                                            \
    wchar_t *end;                                                              \
    long l = (long)func(L"42", &end, 0);                                       \
    assert(l == 42);                                                           \
    assert(*end == L'\0');                                                     \
  }

#define WTESTF(func)                                                           \
  {                                                                            \
    wchar_t *end;                                                              \
    long l = (long)func(L"42", &end);                                          \
    assert(l == 42);                                                           \
    assert(*end == '\0');                                                      \
  }

int main() {
  TESTL(strtol);
  TESTL(strtoll);
  TESTL(strtoimax);
  TESTL(strtoul);
  TESTL(strtoull);
  TESTL(strtoumax);
  TESTF(strtof);
  TESTF(strtod);
  TESTF(strtold);

  WTESTL(wcstol);
  WTESTL(wcstoll);
  WTESTL(wcstoul);
  WTESTL(wcstoull);
  WTESTF(wcstof);
  WTESTF(wcstod);
  WTESTF(wcstold);
}
