// RUN: %clang_asan %s -o %t
// RUN: %env_asan_opts=intercept_strcmp=false %run %t 2>&1
// RUN: %env_asan_opts=intercept_strcmp=true not %run %t 2>&1 | FileCheck %s
// RUN:                                      not %run %t 2>&1 | FileCheck %s

// AIX does not intercept strcmp.
//UNSUPPORTED: target={{.*aix.*}}

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char s1[] = "abcd";
  char s2[] = "1234";
  assert(strcmp(s1, s2) > 0);
  assert(strcmp(s1 - 1, s2));

  // CHECK: {{.*ERROR: AddressSanitizer: stack-buffer-underflow on address}}
  // Very rarely `s1[-1]` happens to be '1', resulting in `strcmp` needing to
  // check 2 bytes before failing, rather than 1 - this should still pass
  // CHECK: READ of size {{[12]}}
  return 0;
}
