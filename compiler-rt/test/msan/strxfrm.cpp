// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t

#include <assert.h>
#include <locale.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  char q[30];
  size_t n = strxfrm(q, "abcdef", sizeof(q));
  assert(n < sizeof(q));
  __msan_check_mem_is_initialized(q, n + 1);

  locale_t loc = newlocale(LC_ALL_MASK, "", (locale_t)0);

  __msan_poison(&q, sizeof(q));
  n = strxfrm_l(q, "qwerty", sizeof(q), loc);
  assert(n < sizeof(q));
  __msan_check_mem_is_initialized(q, n + 1);
  return 0;
}
