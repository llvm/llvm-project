//===-- sanitizer_linux_s390.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements s390-linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_LINUX && SANITIZER_S390

#include "sanitizer_linux.h"

#include <errno.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace __sanitizer {

// --------------- sanitizer_libc.h
uptr internal_mmap(void *addr, uptr length, int prot, int flags, int fd,
                   OFF_T offset) {
  struct s390_mmap_params {
    unsigned long addr;
    unsigned long length;
    unsigned long prot;
    unsigned long flags;
    unsigned long fd;
    unsigned long offset;
  } params = {
    (unsigned long)addr,
    (unsigned long)length,
    (unsigned long)prot,
    (unsigned long)flags,
    (unsigned long)fd,
# ifdef __s390x__
    (unsigned long)offset,
# else
    (unsigned long)(offset / 4096),
# endif
  };
# ifdef __s390x__
  return syscall(__NR_mmap, &params);
# else
  return syscall(__NR_mmap2, &params);
# endif
}

uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  // Minimum frame size.
#ifdef __s390x__
  child_stack = (char *)child_stack - 160;
#else
  child_stack = (char *)child_stack - 96;
#endif
  // Terminate unwind chain.
  ((unsigned long *)child_stack)[0] = 0;
  // And pass parameters.
  ((unsigned long *)child_stack)[1] = (uptr)fn;
  ((unsigned long *)child_stack)[2] = (uptr)arg;
  register long res __asm__("r2");
  register void *__cstack      __asm__("r2") = child_stack;
  register int __flags         __asm__("r3") = flags;
  register int * __ptidptr     __asm__("r4") = parent_tidptr;
  register int * __ctidptr     __asm__("r5") = child_tidptr;
  register void * __newtls     __asm__("r6") = newtls;

  __asm__ __volatile__(
                       /* Clone. */
                       "svc    %1\n"

                       /* if (%r2 != 0)
                        *   return;
                        */
#ifdef __s390x__
                       "cghi   %%r2, 0\n"
#else
                       "chi    %%r2, 0\n"
#endif
                       "jne    1f\n"

                       /* Call "fn(arg)". */
#ifdef __s390x__
                       "lmg    %%r1, %%r2, 8(%%r15)\n"
#else
                       "lm     %%r1, %%r2, 4(%%r15)\n"
#endif
                       "basr   %%r14, %%r1\n"

                       /* Call _exit(%r2). */
                       "svc %2\n"

                       /* Return to parent. */
                     "1:\n"
                       : "=r" (res)
                       : "i"(__NR_clone), "i"(__NR_exit),
                         "r"(__cstack),
                         "r"(__flags),
                         "r"(__ptidptr),
                         "r"(__ctidptr),
                         "r"(__newtls)
                       : "memory", "cc");
  return res;
}

} // namespace __sanitizer

#endif // SANITIZER_LINUX && SANITIZER_S390
