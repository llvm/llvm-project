// RUN: %clangxx_msan %s -o %t
// RUN: %run %t --disable-checks 0 2>&1 | FileCheck --check-prefix=DISABLED --allow-empty %s
// RUN: %run %t --disable-checks 1 2>&1 | FileCheck --check-prefix=DISABLED --allow-empty %s
// RUN: %run %t --disable-checks 2 2>&1 | FileCheck --check-prefix=DISABLED --allow-empty %s
// RUN: %run %t --disable-checks 3 2>&1 | FileCheck --check-prefix=DISABLED --allow-empty %s
// RUN: not %run %t --reenable-checks 0 2>&1 | FileCheck --check-prefix=CASE-0 %s
// RUN: not %run %t --reenable-checks 1 2>&1 | FileCheck --check-prefix=CASE-1 %s
// RUN: not %run %t --reenable-checks 2 2>&1 | FileCheck --check-prefix=CASE-2 %s
// RUN: not %run %t --reenable-checks 3 2>&1 | FileCheck --check-prefix=CASE-3 %s

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  assert(argc == 3);
  __msan_scoped_disable_interceptor_checks();
  if (strcmp(argv[1], "--reenable-checks") == 0)
    __msan_scoped_enable_interceptor_checks();

  char uninit[7];
  switch (argv[2][0]) {
    case '0': {
      char *copy = strndup(uninit, sizeof(uninit));  // BOOM
      free(copy);
      break;
      // CASE-0: Uninitialized bytes in strndup
    }
    case '1': {
      puts(uninit);  // BOOM
      puts(uninit);  // Ensure previous call did not enable interceptor checks.
      break;
      // CASE-1: Uninitialized bytes in puts
    }
    case '2': {
      int cmp = memcmp(uninit, uninit, sizeof(uninit));  // BOOM
      break;
      // CASE-2: Uninitialized bytes in MemcmpInterceptorCommon
    }
    case '3': {
      size_t len = strlen(uninit);  // BOOM
      break;
      // CASE-3: Uninitialized bytes in strlen
    }
    default: assert(0);
  }
  // DISABLED-NOT: Uninitialized bytes
  return 0;
}

