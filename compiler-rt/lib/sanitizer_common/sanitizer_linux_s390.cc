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

} // namespace __sanitizer

#endif // SANITIZER_LINUX && SANITIZER_S390
