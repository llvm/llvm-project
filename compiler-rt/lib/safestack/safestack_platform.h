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

#include "safestack_util.h"
#include "sanitizer_common/sanitizer_platform.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#if !(SANITIZER_NETBSD || SANITIZER_FREEBSD || SANITIZER_LINUX)
#error "Support for your platform has not been implemented"
#endif

#if SANITIZER_NETBSD
#include <lwp.h>

extern "C" void *__mmap(void *, size_t, int, int, int, int, off_t);
#endif

#if SANITIZER_FREEBSD
#include <sys/thr.h>
#endif

namespace safestack {

#if SANITIZER_NETBSD
static void *GetRealLibcAddress(const char *symbol) {
  void *real = dlsym(RTLD_NEXT, symbol);
  if (!real)
    real = dlsym(RTLD_DEFAULT, symbol);
  if (!real) {
    fprintf(stderr, "safestack GetRealLibcAddress failed for symbol=%s",
            symbol);
    abort();
  }
  return real;
}

#define _REAL(func, ...) real##_##func(__VA_ARGS__)
#define DEFINE__REAL(ret_type, func, ...)                              \
  static ret_type (*real_##func)(__VA_ARGS__) = NULL;                  \
  if (!real_##func) {                                                  \
    real_##func = (ret_type(*)(__VA_ARGS__))GetRealLibcAddress(#func); \
  }                                                                    \
  SFS_CHECK(real_##func);
#endif

using ThreadId = uint64_t;

inline ThreadId GetTid() {
#if SANITIZER_NETBSD
  DEFINE__REAL(int, _lwp_self);
  return _REAL(_lwp_self);
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
  DEFINE__REAL(int, _lwp_kill, int a, int b);
  (void)pid;
  return _REAL(_lwp_kill, tid, sig);
#elif SANITIZER_FREEBSD
  return syscall(SYS_thr_kill2, pid, tid, sig);
#else
  return syscall(SYS_tgkill, pid, tid, sig);
#endif
}

inline void *Mmap(void *addr, size_t length, int prot, int flags, int fd,
                  off_t offset) {
#if SANITIZER_NETBSD
  return __mmap(addr, length, prot, flags, fd, 0, offset);
#elif defined(__x86_64__) && (SANITIZER_FREEBSD)
  return (void *)__syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
#else
  return (void *)syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
#endif
}

inline int Munmap(void *addr, size_t length) {
#if SANITIZER_NETBSD
  DEFINE__REAL(int, munmap, void *a, size_t b);
  return _REAL(munmap, addr, length);
#else
  return syscall(SYS_munmap, addr, length);
#endif
}

inline int Mprotect(void *addr, size_t length, int prot) {
#if SANITIZER_NETBSD
  DEFINE__REAL(int, mprotect, void *a, size_t b, int c);
  return _REAL(mprotect, addr, length, prot);
#else
  return syscall(SYS_mprotect, addr, length, prot);
#endif
}

}  // namespace safestack

#endif  // SAFESTACK_PLATFORM_H
