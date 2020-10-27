// RUN: %clangxx_msan -std=c++11 -O0 -g %s -o %t && %run %t
// REQUIRES: x86_64-linux

#include <assert.h>
#include <locale.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>
#include <string.h>

extern "C" decltype(strxfrm_l) __strxfrm_l;

int main(void) {
  char q[100];
  locale_t loc = newlocale(LC_ALL_MASK, "", (locale_t)0);
  size_t n = __strxfrm_l(q, "qwerty", sizeof(q), loc);
  assert(n < sizeof(q));
  __msan_check_mem_is_initialized(q, n + 1);
  return 0;
}
