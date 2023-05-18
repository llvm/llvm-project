// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -g -DUNINIT %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <time.h>

#include <sanitizer/msan_interface.h>

int main(void) {
  struct tm tm;
  tm.tm_year = 2014;
  tm.tm_mon = 3;
  tm.tm_mday = 28;
#ifndef UNINIT
  tm.tm_hour = 13;
#endif
  tm.tm_min = 4;
  tm.tm_sec = 42;
  tm.tm_isdst = -1;
  time_t t = mktime(&tm);
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: in main{{.*}}mktime.cpp:[[@LINE-2]]
  assert(t != -1);
  assert(__msan_test_shadow(&tm, sizeof(tm)) == -1);
  return 0;
}
