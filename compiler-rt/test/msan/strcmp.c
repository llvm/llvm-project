// RUN: %clang_msan %s -o %t
// RUN: env MSAN_OPTIONS=intercept_strcmp=false %run %t 2>&1
// RUN: env MSAN_OPTIONS=intercept_strcmp=true not %run %t 2>&1 | FileCheck %s
// RUN:                                    not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char undef;
  char s1[] = "abcd";
  char s2[] = "1234";
  assert(strcmp(s1, s2) > 0);
  s2[0] = undef;
  assert(strcmp(s1, s2));

  // CHECK: {{.*WARNING: MemorySanitizer: use-of-uninitialized-value}}
  return 0;
}
