// RUN: %clang -O0 %s -o %t && %run %t
// UNSUPPORTED: darwin

#include <assert.h>
#include <locale.h>
#include <wchar.h>

int main(int argc, char **argv) {
  wchar_t q[10];
  size_t n = wcsxfrm(q, L"abcdef", sizeof(q) / sizeof(wchar_t));
  assert(n < sizeof(q));

  wchar_t q2[10];
  locale_t loc = newlocale(LC_ALL_MASK, "", (locale_t)0);
  n = wcsxfrm_l(q2, L"qwerty", sizeof(q) / sizeof(wchar_t), loc);
  assert(n < sizeof(q2));

  freelocale(loc);
  return 0;
}
