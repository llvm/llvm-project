// RUN: %clangxx_msan %s -o %t && not %run %t 2>&1 | FileCheck --check-prefix=ON %s
// RUN: %clangxx_msan %s -o %t && MSAN_OPTIONS=intercept_strndup=0 %run %t 2>&1 | FileCheck --check-prefix=OFF --allow-empty %s

// When built as C on Linux, strndup is transformed to __strndup.
// RUN: %clangxx_msan -O3 -xc %s -o %t && not %run %t 2>&1 | FileCheck --check-prefix=ON %s

// UNSUPPORTED: windows-msvc

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

int main(int argc, char **argv) {
  char kString[4] = "abc";
  __msan_poison(kString + 2, 1);
  char *copy = strndup(kString, 4); // BOOM
  assert(__msan_test_shadow(copy, 4) == 2); // Poisoning is preserved.
  free(copy);
  return 0;
  // ON: Uninitialized bytes in __interceptor_{{(__)?}}strndup at offset 2 inside [{{.*}}, 4)
  // ON: MemorySanitizer: use-of-uninitialized-value
  // ON: #0 {{.*}}main {{.*}}strndup.cpp:[[@LINE-6]]
  // ON-LABEL: SUMMARY
  // ON: {{.*}}strndup.cpp:[[@LINE-8]]
  // OFF-NOT: MemorySanitizer
}

