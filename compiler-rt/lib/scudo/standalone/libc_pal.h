//===-- libc_pal.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_LIBC_PAL_H_
#define SCUDO_LIBC_PAL_H_

#include "platform.h"

#include "internal_defs.h"

#if defined(LIBC_FULL_BUILD)
#include "src/__support/threads/callonce.h"
#include "src/__support/threads/thread.h"
#endif

#if !defined(LIBC_FULL_BUILD) && (SCUDO_LINUX || SCUDO_TRUSTY)
#include <pthread.h>
#endif

#if (SCUDO_LINUX || SCUDO_TRUSTY) &&                                           \
    !defined(SCUDO_USE_LLVM_LIBC_INTERNAL_HEADERS)
#include <errno.h>
#include <string.h>
#include <sys/auxv.h>
#endif

#if SCUDO_LINUX && !defined(SCUDO_USE_LLVM_LIBC_INTERNAL_HEADERS)
#include <fcntl.h>
#include <linux/futex.h>
#include <sched.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#endif

#ifdef SCUDO_USE_LLVM_LIBC_INTERNAL_HEADERS
#include "shared/libc_common.h"
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#endif
#include "src/__support/StringUtil/error_to_string.h"
#include "src/__support/libc_errno.h"
#include "src/__support/threads/identifier.h"
#include "src/__support/time/clock_gettime.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"
#include "src/string/memory_utils/inline_strcmp.h"
#include "src/string/string_length.h"
#if SCUDO_LINUX
#include "hdr/sys_auxv_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/cpu_set_t.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/read.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/write.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/str_to_integer.h"
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#if SCUDO_LINUX || SCUDO_TRUSTY
namespace scudo {

#ifdef SCUDO_USE_LLVM_LIBC_INTERNAL_HEADERS
struct LibcPAL {
private:
  template <typename RetT, typename... Args>
  static RetT doSyscall(long Number, Args... args) {
    RetT Ret = LIBC_NAMESPACE::syscall_impl<RetT>(Number, args...);
    if (static_cast<unsigned long>(Ret) > -4096UL) {
      seterrno(-static_cast<int>(Ret));
      return static_cast<RetT>(-1);
    }
    return Ret;
  }

  template <typename T> static T unwrapErrorOr(LIBC_NAMESPACE::ErrorOr<T> Ret) {
    if (!Ret.has_value()) {
      seterrno(Ret.error());
      return static_cast<T>(-1);
    }
    return Ret.value();
  }

public:
  static constexpr uptr kMmapFailed = static_cast<uptr>(-1);
#if defined(LIBC_FULL_BUILD)
  using PThreadKeyT = unsigned int;
  using PThreadOnceT = LIBC_NAMESPACE::CallOnceFlag;
#else
  using PThreadKeyT = pthread_key_t;
  using PThreadOnceT = pthread_once_t;
#endif
  static int geterrno() { return libc_errno; }
  static void seterrno(int ErrorNum) { libc_errno = ErrorNum; }
  static long strtol(const char *Str, char **EndPtr, int Base) {
    auto result = LIBC_NAMESPACE::internal::strtointeger<long>(Str, Base);
    if (result.has_error())
      libc_errno = result.error;

    if (EndPtr != nullptr)
      *EndPtr = const_cast<char *>(Str + result.parsed_len);

    return result;
  }
  static unsigned long getauxval(unsigned long Type) {
    LIBC_NAMESPACE::cpp::optional<unsigned long> Data =
        LIBC_NAMESPACE::auxv::get(Type);
    return Data.has_value() ? Data.value() : 0;
  }
  static char *strerror(int ErrorNum) {
    return const_cast<char *>(
        LIBC_NAMESPACE::get_error_string(ErrorNum).data());
  }
  static size_t strlen(const char *Str) {
    return LIBC_NAMESPACE::internal::string_length(Str);
  }
  static int strncmp(const char *LHS, const char *RHS, size_t Count) {
    auto Comp = [](char L, char R) -> int {
      return static_cast<unsigned char>(L) -
             static_cast<unsigned char>(R);
    };
    return LIBC_NAMESPACE::inline_strncmp(LHS, RHS, Count, Comp);
  }
  static void *memcpy(void *Dst, const void *Src, size_t Count) {
    LIBC_NAMESPACE::inline_memcpy(Dst, Src, Count);
    return Dst;
  }
  static void *memset(void *Dst, int Value, size_t Count) {
    LIBC_NAMESPACE::inline_memset(Dst, static_cast<u8>(Value), Count);
    return Dst;
  }
  static int pthread_key_create(PThreadKeyT *Key, void (*Dtor)(void *)) {
#if defined(LIBC_FULL_BUILD)
    auto NewKey = LIBC_NAMESPACE::new_tss_key(Dtor);
    if (!NewKey.has_value())
      return EINVAL;
    *Key = NewKey.value();
    return 0;
#else
    return ::pthread_key_create(Key, Dtor);
#endif
  }
  static int pthread_key_delete(PThreadKeyT Key) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::tss_key_delete(Key) ? 0 : EINVAL;
#else
    return ::pthread_key_delete(Key);
#endif
  }
  static void *pthread_getspecific(PThreadKeyT Key) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::get_tss_value(Key);
#else
    return ::pthread_getspecific(Key);
#endif
  }
  static int pthread_setspecific(PThreadKeyT Key, const void *Value) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::set_tss_value(Key, const_cast<void *>(Value)) ? 0
                                                                         : EINVAL;
#else
    return ::pthread_setspecific(Key, Value);
#endif
  }
#if defined(LIBC_FULL_BUILD)
  static constexpr PThreadOnceT initPThreadOnce() {
    return PThreadOnceT(LIBC_NAMESPACE::callonce_impl::NOT_CALLED);
  }
#else
  static PThreadOnceT initPThreadOnce() { return PTHREAD_ONCE_INIT; }
#endif
  static int pthread_once(PThreadOnceT *Flag, void (*Func)(void)) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::callonce(Flag, Func);
#else
    return ::pthread_once(Flag, Func);
#endif
  }

#if SCUDO_LINUX
  static void abort() {
    LIBC_NAMESPACE::write_to_stderr("aborted due to scudo allocator error\n");
    LIBC_NAMESPACE::internal::exit(127);
  }
  static void *mmap(void *Addr, uptr Size, int Prot, int Flags, int FD,
                    off_t Offset) {
    long MmapOffset = static_cast<long>(Offset);
#ifdef SYS_mmap2
    MmapOffset /= 4096;
    long Ret = LIBC_NAMESPACE::syscall_impl<long>(
        SYS_mmap2, reinterpret_cast<long>(Addr), Size, Prot, Flags, FD,
        MmapOffset);
#elif defined(SYS_mmap)
    long Ret = LIBC_NAMESPACE::syscall_impl<long>(
        SYS_mmap, reinterpret_cast<long>(Addr), Size, Prot, Flags, FD,
        MmapOffset);
#else
#error "mmap or mmap2 syscalls not available."
#endif
    if (!LIBC_NAMESPACE::linux_utils::is_valid_mmap(Ret)) {
      seterrno(-static_cast<int>(Ret));
      return reinterpret_cast<void *>(kMmapFailed);
    }
    return reinterpret_cast<void *>(Ret);
  }
  static int munmap(void *Addr, uptr Size) {
    return doSyscall<int>(SYS_munmap, reinterpret_cast<long>(Addr), Size);
  }
  static int mprotect(void *Addr, uptr Size, int Prot) {
    return doSyscall<int>(SYS_mprotect, reinterpret_cast<long>(Addr), Size,
                          Prot);
  }
  static int madvise(void *Addr, uptr Size, int Advice) {
    return doSyscall<int>(SYS_madvise, reinterpret_cast<long>(Addr), Size,
                          Advice);
  }
  static char *getenv(const char *Name) {
    (void)Name;
    return nullptr;
  }
  template <typename... Args>
  static long systemCall(long Number, Args... args) {
    return doSyscall<long>(Number, args...);
  }
  static int clock_gettime(clockid_t ClockID, timespec *TS) {
    LIBC_NAMESPACE::ErrorOr<int> Ret =
        LIBC_NAMESPACE::internal::clock_gettime(ClockID, TS);
    if (!Ret.has_value()) {
      seterrno(Ret.error());
      return -1;
    }
    return Ret.value();
  }
  static int sched_getaffinity(pid_t TID, size_t CpuSetSize, cpu_set_t *Mask) {
    int Ret = LIBC_NAMESPACE::syscall_impl<int>(SYS_sched_getaffinity, TID,
                                                CpuSetSize, Mask);
    if (Ret < 0) {
      seterrno(-Ret);
      return -1;
    }
    if (size_t(Ret) < CpuSetSize) {
      u8 *MaskBytes = reinterpret_cast<u8 *>(Mask);
      for (size_t I = size_t(Ret); I < CpuSetSize; ++I)
        MaskBytes[I] = 0;
    }
    return 0;
  }
  static int cpuCount(const cpu_set_t *Mask) {
    int Count = 0;
    for (uptr I = 0; I < ARRAY_SIZE(Mask->__mask); ++I)
      Count += __builtin_popcountl(Mask->__mask[I]);
    return Count;
  }
  static pid_t gettid() { return LIBC_NAMESPACE::internal::gettid(); }
  static int open(const char *Path, int Flags) {
    return unwrapErrorOr(LIBC_NAMESPACE::linux_syscalls::open(Path, Flags, 0));
  }
  static ssize_t read(int FD, void *Buffer, size_t Count) {
    return unwrapErrorOr(
        LIBC_NAMESPACE::linux_syscalls::read(FD, Buffer, Count));
  }
  static int close(int FD) {
    return unwrapErrorOr(LIBC_NAMESPACE::linux_syscalls::close(FD));
  }
  static ssize_t write(int FD, const void *Buffer, size_t Count) {
    return unwrapErrorOr(
        LIBC_NAMESPACE::linux_syscalls::write(FD, Buffer, Count));
  }
  static int mincore(void *Addr, uptr Size, unsigned char *Vec) {
    long Ret = LIBC_NAMESPACE::syscall_impl<long>(
        SYS_mincore, reinterpret_cast<long>(Addr), Size,
        reinterpret_cast<long>(Vec));
    if (Ret < 0) {
      seterrno(-static_cast<int>(Ret));
      return -1;
    }
    return 0;
  }
  static int prctl(int Option, int Arg2, void *Arg3, uptr Arg4,
                   const char *Arg5) {
    return doSyscall<int>(SYS_prctl, Option, Arg2, Arg3, Arg4, Arg5);
  }
#endif
};
#else
struct LibcPAL {
  static constexpr uptr kMmapFailed = static_cast<uptr>(-1);
#if defined(LIBC_FULL_BUILD)
  using PThreadKeyT = unsigned int;
  using PThreadOnceT = LIBC_NAMESPACE::CallOnceFlag;
#else
  using PThreadKeyT = pthread_key_t;
  using PThreadOnceT = pthread_once_t;
#endif
  static int geterrno() { return errno; }
  static void seterrno(int ErrorNum) { errno = ErrorNum; }
  static long strtol(const char *Str, char **EndPtr, int Base) {
    return ::strtol(Str, EndPtr, Base);
  }
  static unsigned long getauxval(unsigned long Type) {
    return ::getauxval(Type);
  }
  static char *strerror(int ErrorNum) { return ::strerror(ErrorNum); }
  static size_t strlen(const char *Str) { return ::strlen(Str); }
  static int strncmp(const char *LHS, const char *RHS, size_t Count) {
    return ::strncmp(LHS, RHS, Count);
  }
  static void *memcpy(void *Dst, const void *Src, size_t Count) {
    return ::memcpy(Dst, Src, Count);
  }
  static void *memset(void *Dst, int Value, size_t Count) {
    return ::memset(Dst, Value, Count);
  }
  static int pthread_key_create(PThreadKeyT *Key, void (*Dtor)(void *)) {
#if defined(LIBC_FULL_BUILD)
    auto NewKey = LIBC_NAMESPACE::new_tss_key(Dtor);
    if (!NewKey.has_value())
      return EINVAL;
    *Key = NewKey.value();
    return 0;
#else
    return ::pthread_key_create(Key, Dtor);
#endif
  }
  static int pthread_key_delete(PThreadKeyT Key) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::tss_key_delete(Key) ? 0 : EINVAL;
#else
    return ::pthread_key_delete(Key);
#endif
  }
  static void *pthread_getspecific(PThreadKeyT Key) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::get_tss_value(Key);
#else
    return ::pthread_getspecific(Key);
#endif
  }
  static int pthread_setspecific(PThreadKeyT Key, const void *Value) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::set_tss_value(Key, const_cast<void *>(Value)) ? 0
                                                                         : EINVAL;
#else
    return ::pthread_setspecific(Key, Value);
#endif
  }
  static PThreadOnceT initPThreadOnce() {
#if defined(LIBC_FULL_BUILD)
    return PThreadOnceT(LIBC_NAMESPACE::callonce_impl::NOT_CALLED);
#else
    return PTHREAD_ONCE_INIT;
#endif
  }
  static int pthread_once(PThreadOnceT *Flag, void (*Func)(void)) {
#if defined(LIBC_FULL_BUILD)
    return LIBC_NAMESPACE::callonce(Flag, Func);
#else
    return ::pthread_once(Flag, Func);
#endif
  }

#if SCUDO_LINUX
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
  static long systemCall(long Number) { return ::syscall(Number); }
  static long systemCall(long Number, uptr Arg1, int Arg2, u32 Arg3,
                         void *Arg4, void *Arg5, u32 Arg6) {
    return ::syscall(Number, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6);
  }
  static long systemCall(long Number, void *Arg1, uptr Arg2, unsigned Arg3) {
    return ::syscall(Number, Arg1, Arg2, Arg3);
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
#endif

} // namespace scudo
#endif

#endif // SCUDO_LIBC_PAL_H_
