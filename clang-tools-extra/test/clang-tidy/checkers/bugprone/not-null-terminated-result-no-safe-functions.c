// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-not-null-terminated-result.WantToUseSafeFunctions: false \
// RUN:   }}" -- -I %S/Inputs/not-null-terminated-result

#include "not-null-terminated-result-c.h"

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

void test_memcpy_no_safe(const char *src) {
  char dest[13];
  memcpy(dest, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest[14];
  // CHECK-FIXES-NEXT: strcpy(dest, src);
}
