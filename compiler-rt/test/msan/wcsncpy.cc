// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 %s -o %t && not %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out && FileCheck %s < %t.out

// XFAIL: mips

#include <assert.h>
#include <wchar.h>

#include <sanitizer/msan_interface.h>

int main() {
  const wchar_t *s = L"abc";
  assert(wcslen(s) == 3);

  wchar_t s2[5];
  assert(wcsncpy(s2, s, 3) == s2);
  assert(__msan_test_shadow(&s2, 5 * sizeof(wchar_t)) == 3 * sizeof(wchar_t));
  assert(wcsncpy(s2, s, 5) == s2);
  assert(__msan_test_shadow(&s2, 5 * sizeof(wchar_t)) == -1);

  wchar_t s3[5];
  assert(wcsncpy(s3, s, 2) == s3);
  assert(__msan_test_shadow(&s3, 5 * sizeof(wchar_t)) == 2 * sizeof(wchar_t));

  __msan_allocated_memory(&s2[1], sizeof(wchar_t));
  wchar_t s4[5];
  assert(wcsncpy(s4, s2, 3) == s4);
  __msan_check_mem_is_initialized(&s4, sizeof(s4));
}
// CHECK:  Uninitialized bytes in __msan_check_mem_is_initialized
// CHECK:  WARNING: MemorySanitizer: use-of-uninitialized-value
// CHECK:    in main {{.*}}wcsncpy.cc:28

// CHECK:  Uninitialized value was stored to memory at
// CHECK:    in {{[^\s]*}}wcsncpy
// CHECK:    in main {{.*}}wcsncpy.cc:27

// CHECK:  Memory was marked as uninitialized
// CHECK:    in __msan_allocated_memory
// CHECK:    in main {{.*}}wcsncpy.cc:25
