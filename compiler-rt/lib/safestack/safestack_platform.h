//===-- safestack_platform.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements platform specific parts of SafeStack runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SAFESTACK_PLATFORM_H
#define SAFESTACK_PLATFORM_H

#include "sanitizer_common/sanitizer_platform.h"

#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#if !(SANITIZER_NETBSD || SANITIZER_FREEBSD || SANITIZER_LINUX)
#error "Support for your platform has not been implemented"
#endif

#if SANITIZER_NETBSD
#include <lwp.h>
#endif

#if SANITIZER_FREEBSD
#include <sys/thr.h>
#endif

namespace safestack {

using ThreadId = uint64_t;

inline ThreadId GetTid() {
#if SANITIZER_NETBSD
  return _lwp_self();
#elif SANITIZER_FREEBSD
  long Tid;
  thr_self(&Tid);
  return Tid;
#else
  return syscall(SYS_gettid);
#endif
}

inline int TgKill(pid_t pid, ThreadId tid, int sig) {
#if SANITIZER_NETBSD
  (void)pid;
  return _lwp_kill(tid, sig);
#elif SANITIZER_FREEBSD
  return syscall(SYS_thr_kill2, pid, tid, sig);
#else
  return syscall(SYS_tgkill, pid, tid, sig);
#endif
}

inline void *Mmap(void *addr, size_t length, int prot, int flags, int fd,
                  off_t offset) {
#if SANITIZER_NETBSD
  return mmap(addr, length, prot, flags, fd, offset);
#elif defined(__x86_64__) && (SANITIZER_FREEBSD)
  return (void *)__syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
#else
  return (void *)syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
#endif
}

inline int Munmap(void *addr, size_t length) {
#if SANITIZER_NETBSD
  return munmap(addr, length);
#else
  return syscall(SYS_munmap, addr, length);
#endif
}

inline int Mprotect(void *addr, size_t length, int prot) {
#if SANITIZER_NETBSD
  return mprotect(addr, length, prot);
#else
  return syscall(SYS_mprotect, addr, length, prot);
#endif
}

}  // namespace safestack

#endif  // SAFESTACK_PLATFORM_H
