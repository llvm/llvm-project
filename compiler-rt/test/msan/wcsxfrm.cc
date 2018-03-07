// RUN: %clangxx_msan -O0 -g %s -o %t && not %run %t

#include <assert.h>
#include <locale.h>
#include <sanitizer/msan_interface.h>
#include <stdlib.h>
#include <wchar.h>

int main(void) {
  wchar_t q[10];
  size_t n = wcsxfrm(q, L"abcdef", sizeof(q) / sizeof(wchar_t));
  assert(n < sizeof(q));
  __msan_check_mem_is_initialized(q, (n + 1) * sizeof(wchar_t));

  locale_t loc = newlocale(LC_ALL_MASK, "", (locale_t)0);

  __msan_poison(&q, sizeof(q));
  n = wcsxfrm_l(q, L"qwerty", sizeof(q) / sizeof(wchar_t), loc);
  assert(n < sizeof(q));
  __msan_check_mem_is_initialized(q, (n + 1) * sizeof(wchar_t));

  q[0] = 'A';
  q[1] = '\x00';
  __msan_poison(&q, sizeof(q));
  wcsxfrm(NULL, q, 0);

  // CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // CHECK:    in main {{.*}}wcsxfrm.cc:25
  return 0;
}
