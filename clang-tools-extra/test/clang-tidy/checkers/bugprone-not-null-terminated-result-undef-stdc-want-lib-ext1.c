// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -std=c11 -I %S/Inputs/bugprone-not-null-terminated-result

#include "not-null-terminated-result-c.h"

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1
#undef __STDC_WANT_LIB_EXT1__

void f(const char *src) {
  char dest[13];
  memcpy_s(dest, 13, src, strlen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest[14];
  // CHECK-FIXES-NEXT: strncpy_s(dest, 14, src, strlen(src) - 1);
}

