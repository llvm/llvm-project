// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-apple-darwin9
// expected-no-diagnostics

// From <rdar://problem/12322000>.  Do not warn about undefined behavior of parameter
// argument types in unreachable code in a macro.
#define VA_ARG_RDAR12322000(Marker, TYPE)         ((sizeof (TYPE) < sizeof (UINTN_RDAR12322000)) ? (TYPE)(__builtin_va_arg (Marker, UINTN_RDAR12322000)) : (TYPE)(__builtin_va_arg (Marker, TYPE)))

// 64-bit system
typedef unsigned long long  UINTN_RDAR12322000;

int test_VA_ARG_RDAR12322000 (__builtin_va_list Marker)
{
  return VA_ARG_RDAR12322000 (Marker, short); // no-warning
}