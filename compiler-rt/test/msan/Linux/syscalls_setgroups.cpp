// RUN: %clangxx_msan -DPRE1 -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <string.h>

#include <sanitizer/linux_syscall_hooks.h>
#include <sanitizer/msan_interface.h>

// The setgroups pre-hook must READ (check) its input group list, not write to
// (unpoison) it: grouplist is a pure input the kernel only reads. Passing an
// uninitialized list is a caller-side bug the pre-hook must report.
int main() {
  unsigned int groups[4];

#if defined(PRE1)
  // Uninitialized input -> the pre-hook must report use-of-uninitialized-value.
  __msan_poison(groups, sizeof(groups));
  __sanitizer_syscall_pre_setgroups(4, groups);
  // CHECK: MemorySanitizer: use-of-uninitialized-value
#else
  // A fully-initialized input is fine: no report, and the hook must not
  // unpoison caller memory.
  memset(groups, 0, sizeof(groups));
  __sanitizer_syscall_pre_setgroups(4, groups);
  __sanitizer_syscall_post_setgroups(0, 4, groups);
  // ...and a clean input stays clean: the hooks must not corrupt its shadow.
  assert(__msan_test_shadow(groups, sizeof(groups)) == -1);
#endif
  return 0;
}
