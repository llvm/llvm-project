// RUN: %clangxx_msan -DPRE1 -O0 %s -o %t && not %run %t 2>&1
// RUN: %clangxx_msan -DPRE2 -O0 %s -o %t && not %run %t 2>&1
// RUN: %clangxx_msan -DPRE3 -O0 %s -o %t && not %run %t 2>&1
// RUN: %clangxx_msan -O0 %s -o %t && %run %t 2>&1

#include <assert.h>
#include <signal.h>
#include <string.h>

#include <sanitizer/linux_syscall_hooks.h>
#include <sanitizer/msan_interface.h>

struct my_kernel_sigaction {
#if defined(__mips__)
  long flags, handler;
#else
  long handler, flags, restorer;
#endif
  uint64_t mask[20]; // larger than any known platform
};

int main() {
  my_kernel_sigaction act = {}, oldact = {};

#if defined(PRE1)
  __msan_poison(&act.handler, sizeof(act.handler));
  __sanitizer_syscall_pre_rt_sigaction(SIGUSR1, &act, &oldact, 20 * 8);
#elif defined(PRE2)
  __msan_poison(&act.flags, sizeof(act.flags));
  __sanitizer_syscall_pre_rt_sigaction(SIGUSR1, &act, &oldact, 20 * 8);
#elif defined(PRE3)
  __msan_poison(&act.mask, 1);
  __sanitizer_syscall_pre_rt_sigaction(SIGUSR1, &act, &oldact, 20 * 8);
#else
  // Uninit past the end of the mask is ignored.
  __msan_poison(((char *)&act.mask) + 5, 1);
  __sanitizer_syscall_pre_rt_sigaction(SIGUSR1, &act, &oldact, 5);

  memset(&act, 0, sizeof(act));
  __msan_poison(&oldact, sizeof(oldact));
  __sanitizer_syscall_post_rt_sigaction(0, SIGUSR1, &act, &oldact, 5);
#if defined(__mips__)
  assert(__msan_test_shadow(&oldact, sizeof(oldact)) == sizeof(long)*2 + 5);
#else
  assert(__msan_test_shadow(&oldact, sizeof(oldact)) == sizeof(long)*3 + 5);
#endif
#endif
}
