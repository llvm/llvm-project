//===-- libc_pal.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_LIBC_PAL_H_
#define SCUDO_LIBC_PAL_H_

#include "internal_defs.h"
#include "platform.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if SCUDO_LINUX || SCUDO_TRUSTY
#include <sys/auxv.h>
#endif

#if SCUDO_LINUX
#include <fcntl.h>
#include <linux/futex.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace scudo {

struct LibcPAL {
  using PThreadKeyT = pthread_key_t;

  static int geterrno() { return errno; }
  static void seterrno(int ErrorNum) { errno = ErrorNum; }

  static long strtol(const char *Str, char **EndPtr, int Base) {
    return ::strtol(Str, EndPtr, Base);
  }

#if SCUDO_LINUX || SCUDO_TRUSTY
  static unsigned long getauxval(unsigned long Type) {
    return ::getauxval(Type);
  }
#endif

  static char *strerror(int ErrorNum) { return ::strerror(ErrorNum); }
  static size_t strlen(const char *Str) { return ::strlen(Str); }

  static void *memcpy(void *Dst, const void *Src, size_t Count) {
    return ::memcpy(Dst, Src, Count);
  }

  static void *memset(void *Dst, int Value, size_t Count) {
    return ::memset(Dst, Value, Count);
  }

  static int pthread_key_create(PThreadKeyT *Key, void (*Dtor)(void *)) {
    return ::pthread_key_create(Key, Dtor);
  }

  static int pthread_key_delete(PThreadKeyT Key) {
    return ::pthread_key_delete(Key);
  }

  static void *pthread_getspecific(PThreadKeyT Key) {
    return ::pthread_getspecific(Key);
  }

  static int pthread_setspecific(PThreadKeyT Key, const void *Value) {
    return ::pthread_setspecific(Key, Value);
  }

#if SCUDO_LINUX
  static constexpr uptr kMmapFailed = static_cast<uptr>(-1);

  static void abort() { ::abort(); }

  static void *mmap(void *Addr, uptr Size, int Prot, int Flags, int FD,
                    off_t Offset) {
    return ::mmap(Addr, Size, Prot, Flags, FD, Offset);
  }

  static int munmap(void *Addr, uptr Size) { return ::munmap(Addr, Size); }

  static int mprotect(void *Addr, uptr Size, int Prot) {
    return ::mprotect(Addr, Size, Prot);
  }

  static int madvise(void *Addr, uptr Size, int Advice) {
    return ::madvise(Addr, Size, Advice);
  }

  static char *getenv(const char *Name) { return ::getenv(Name); }

  template <typename... Args> static long syscall(long Number, Args... args) {
    return ::syscall(Number, args...);
  }

  static int clock_gettime(clockid_t ClockID, timespec *TS) {
    return ::clock_gettime(ClockID, TS);
  }

  static int sched_getaffinity(pid_t PID, size_t CpuSetSize, cpu_set_t *Mask) {
    return ::sched_getaffinity(PID, CpuSetSize, Mask);
  }

  static int cpuCount(const cpu_set_t *Mask) { return CPU_COUNT(Mask); }
  static pid_t gettid() { return ::gettid(); }
  static int open(const char *Path, int Flags) { return ::open(Path, Flags); }

  static ssize_t read(int FD, void *Buffer, size_t Count) {
    return ::read(FD, Buffer, Count);
  }

  static int close(int FD) { return ::close(FD); }

  static ssize_t write(int FD, const void *Buffer, size_t Count) {
    return ::write(FD, Buffer, Count);
  }

  static int mincore(void *Addr, uptr Size, unsigned char *Vec) {
    return ::mincore(Addr, Size, Vec);
  }

  static int prctl(int Option, int Arg2, void *Arg3, uptr Arg4,
                   const char *Arg5) {
    return ::prctl(Option, Arg2, Arg3, Arg4, Arg5);
  }
#endif
};

} // namespace scudo

#endif // SCUDO_LIBC_PAL_H_
