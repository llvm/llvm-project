// RUN: %clang_scudo %s -o %t
// RUN:                                                                                                  %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-nolimit
// RUN: %env_scudo_opts="soft_rss_limit_mb=256"                                                          %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-nolimit
// RUN: %env_scudo_opts="hard_rss_limit_mb=256"                                                          %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-nolimit
// RUN: %env_scudo_opts="soft_rss_limit_mb=64:allocator_may_return_null=0"                           not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-softlimit
// RUN: %env_scudo_opts="soft_rss_limit_mb=64:allocator_may_return_null=1"                               %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-softlimit-returnnull
// RUN: %env_scudo_opts="soft_rss_limit_mb=64:allocator_may_return_null=0:can_use_proc_maps_statm=0" not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-softlimit
// RUN: %env_scudo_opts="soft_rss_limit_mb=64:allocator_may_return_null=1:can_use_proc_maps_statm=0"     %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-softlimit-returnnull
// RUN: %env_scudo_opts="hard_rss_limit_mb=64:allocator_may_return_null=0"                           not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-hardlimit
// RUN: %env_scudo_opts="hard_rss_limit_mb=64:allocator_may_return_null=1"                           not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-hardlimit
// RUN: %env_scudo_opts="hard_rss_limit_mb=64:allocator_may_return_null=0:can_use_proc_maps_statm=0" not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-hardlimit
// RUN: %env_scudo_opts="hard_rss_limit_mb=64:allocator_may_return_null=1:can_use_proc_maps_statm=0" not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-hardlimit

// Tests that the soft and hard RSS limits work as intended. Without limit or
// with a high limit, the test should pass without any malloc returning NULL or
// the program dying.
// If a limit is specified, it should return some NULL or die depending on
// allocator_may_return_null. This should also work without statm.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const size_t kNumAllocs = 128;
static const size_t kAllocSize = 1 << 20;  // 1MB.

static void *allocs[kNumAllocs];

int main(int argc, char *argv[]) {
  int returned_null = 0;
  for (int i = 0; i < kNumAllocs; i++) {
    if ((i & 0xf) == 0)
      usleep(50000);
    allocs[i] = malloc(kAllocSize);
    if (allocs[i])
      memset(allocs[i], 0xff, kAllocSize);  // Dirty the pages.
    else
      returned_null++;
  }
  for (int i = 0; i < kNumAllocs; i++)
    free(allocs[i]);
  if (returned_null == 0)
    printf("All malloc calls succeeded\n");
  else
    printf("%d malloc calls returned NULL\n", returned_null);
  return 0;
}

// CHECK-nolimit: All malloc calls succeeded
// CHECK-softlimit: soft RSS limit exhausted
// CHECK-softlimit-NOT: malloc calls
// CHECK-softlimit-returnnull: soft RSS limit exhausted
// CHECK-softlimit-returnnull: malloc calls returned NULL
// CHECK-hardlimit: hard RSS limit exhausted
// CHECK-hardlimit-NOT: malloc calls
