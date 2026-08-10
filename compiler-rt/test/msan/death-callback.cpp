// RUN: %clangxx_msan -DERROR %s -o %t && not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOCB
// RUN: %clangxx_msan -DERROR -DMSANCB_SET %s -o %t && not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CB
// RUN: %clangxx_msan -DERROR -DMSANCB_SET -DMSANCB_CLEAR %s -o %t && not %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOCB
// RUN: %clangxx_msan -DMSANCB_SET %s -o %t && %run %t 2>&1 | \
// RUN:     FileCheck %s --check-prefixes=SUCCEED,CHECK-NOCB

#include <sanitizer/msan_interface.h>
#include <stdio.h>
#include <stdlib.h>

void cb(void) {
  fprintf(stderr, "msan-death-callback\n");
}

int main(int argc, char **argv) {
  int *volatile p = (int *)malloc(sizeof(int));
  *p = 42;
  free(p);

#ifdef MSANCB_SET
  __msan_set_death_callback(cb);
#endif

#ifdef MSANCB_CLEAR
  __msan_set_death_callback(0);
#endif

#ifdef ERROR
  if (*p)
    exit(0);
#endif
  // CHECK-CB: msan-death-callback
  // CHECK-NOCB-NOT: msan-death-callback

  // CHECK-NOT: done
  // SUCCEED: done
  fprintf(stderr, "done\n");
  return 0;
}
