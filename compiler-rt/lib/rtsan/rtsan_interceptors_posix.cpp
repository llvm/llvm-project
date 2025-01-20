//===--- rtsan_interceptors.cpp - Realtime Sanitizer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "rtsan/rtsan_interceptors.h"

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_dlsym.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"

#include "interception/interception.h"
#include "rtsan/rtsan.h"

#if SANITIZER_APPLE

#if TARGET_OS_MAC
// On MacOS OSSpinLockLock is deprecated and no longer present in the headers,
// but the symbol still exists on the system. Forward declare here so we
// don't get compilation errors.
#include <stdint.h>
extern "C" {
typedef int32_t OSSpinLock;
void OSSpinLockLock(volatile OSSpinLock *__lock);
}
#endif // TARGET_OS_MAC

#include <libkern/OSAtomic.h>
#include <os/lock.h>
#endif // SANITIZER_APPLE

#if SANITIZER_INTERCEPT_MEMALIGN || SANITIZER_INTERCEPT_PVALLOC
#include <malloc.h>
#endif

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

using namespace __sanitizer;

namespace {
struct DlsymAlloc : public DlSymAllocator<DlsymAlloc> {
  static bool UseImpl() { return !__rtsan_is_initialized(); }
};
} // namespace

// Filesystem

INTERCEPTOR(int, open, const char *path, int oflag, ...) {
  // We do not early exit if O_NONBLOCK is set.
  // O_NONBLOCK **does not prevent the syscall** it simply sets the FD to be in
  // nonblocking mode, which is a different concept than our
  // [[clang::nonblocking]], and is not rt-safe. This behavior was confirmed
  // using Instruments on Darwin with a simple test program
  __rtsan_notify_intercepted_call("open");

  if (OpenReadsVaArgs(oflag)) {
    va_list args;
    va_start(args, oflag);
    const mode_t mode = va_arg(args, int);
    va_end(args);
    return REAL(open)(path, oflag, mode);
  }

  return REAL(open)(path, oflag);
}

#if SANITIZER_INTERCEPT_OPEN64
INTERCEPTOR(int, open64, const char *path, int oflag, ...) {
  // See comment above about O_NONBLOCK
  __rtsan_notify_intercepted_call("open64");

  if (OpenReadsVaArgs(oflag)) {
    va_list args;
    va_start(args, oflag);
    const mode_t mode = va_arg(args, int);
    va_end(args);
    return REAL(open64)(path, oflag, mode);
  }

  return REAL(open64)(path, oflag);
}
#define RTSAN_MAYBE_INTERCEPT_OPEN64 INTERCEPT_FUNCTION(open64)
#else
#define RTSAN_MAYBE_INTERCEPT_OPEN64
#endif // SANITIZER_INTERCEPT_OPEN64

INTERCEPTOR(int, openat, int fd, const char *path, int oflag, ...) {
  // See comment above about O_NONBLOCK
  __rtsan_notify_intercepted_call("openat");

  if (OpenReadsVaArgs(oflag)) {
    va_list args;
    va_start(args, oflag);
    const mode_t mode = va_arg(args, int);
    va_end(args);
    return REAL(openat)(fd, path, oflag, mode);
  }

  return REAL(openat)(fd, path, oflag);
}

#if SANITIZER_INTERCEPT_OPENAT64
INTERCEPTOR(int, openat64, int fd, const char *path, int oflag, ...) {
  // See comment above about O_NONBLOCK
  __rtsan_notify_intercepted_call("openat64");

  if (OpenReadsVaArgs(oflag)) {
    va_list args;
    va_start(args, oflag);
    const mode_t mode = va_arg(args, int);
    va_end(args);
    return REAL(openat64)(fd, path, oflag, mode);
  }

  return REAL(openat64)(fd, path, oflag);
}
#define RTSAN_MAYBE_INTERCEPT_OPENAT64 INTERCEPT_FUNCTION(openat64)
#else
#define RTSAN_MAYBE_INTERCEPT_OPENAT64
#endif // SANITIZER_INTERCEPT_OPENAT64

INTERCEPTOR(int, creat, const char *path, mode_t mode) {
  // See comment above about O_NONBLOCK
  __rtsan_notify_intercepted_call("creat");
  const int result = REAL(creat)(path, mode);
  return result;
}

#if SANITIZER_INTERCEPT_CREAT64
INTERCEPTOR(int, creat64, const char *path, mode_t mode) {
  // See comment above about O_NONBLOCK
  __rtsan_notify_intercepted_call("creat64");
  const int result = REAL(creat64)(path, mode);
  return result;
}
#define RTSAN_MAYBE_INTERCEPT_CREAT64 INTERCEPT_FUNCTION(creat64)
#else
#define RTSAN_MAYBE_INTERCEPT_CREAT64
#endif // SANITIZER_INTERCEPT_CREAT64

INTERCEPTOR(int, fcntl, int filedes, int cmd, ...) {
  __rtsan_notify_intercepted_call("fcntl");

  // Following precedent here. The linux source (fcntl.c, do_fcntl) accepts the
  // final argument in a variable that will hold the largest of the possible
  // argument types. It is then assumed that the implementation of fcntl will
  // cast it properly depending on cmd.
  //
  // The two types we expect for possible args are `struct flock*` and `int`
  // we will cast to `intptr_t` which should hold both comfortably.
  // Why `intptr_t`? It should fit both types, and it follows the freeBSD
  // approach linked below.
  using arg_type = intptr_t;
  static_assert(sizeof(arg_type) >= sizeof(struct flock *));
  static_assert(sizeof(arg_type) >= sizeof(int));

  // Some cmds will not actually have an argument passed in this va_list.
  // Calling va_arg when no arg exists is UB, however all currently
  // supported architectures will give us a result in all three cases
  // (no arg/int arg/struct flock* arg)
  // va_arg() will generally read the next argument register or the
  // stack. If we ever support an arch like CHERI with bounds checking, we
  // may have to re-evaluate this approach.
  //
  // More discussion, and other examples following this approach
  // https://discourse.llvm.org/t/how-to-write-an-interceptor-for-fcntl/81203
  // https://reviews.freebsd.org/D46403
  // https://github.com/bminor/glibc/blob/c444cc1d8335243c5c4e636d6a26c472df85522c/sysdeps/unix/sysv/linux/fcntl64.c#L37-L46

  va_list args;
  va_start(args, cmd);
  const arg_type arg = va_arg(args, arg_type);
  va_end(args);

  return REAL(fcntl)(filedes, cmd, arg);
}

INTERCEPTOR(int, ioctl, int filedes, unsigned long request, ...) {
  __rtsan_notify_intercepted_call("ioctl");

  // See fcntl for discussion on why we use intptr_t
  // And why we read from va_args on all request types
  using arg_type = intptr_t;
  static_assert(sizeof(arg_type) >= sizeof(struct ifreq *));
  static_assert(sizeof(arg_type) >= sizeof(int));

  va_list args;
  va_start(args, request);
  arg_type arg = va_arg(args, arg_type);
  va_end(args);

  return REAL(ioctl)(filedes, request, arg);
}

#if SANITIZER_INTERCEPT_FCNTL64
INTERCEPTOR(int, fcntl64, int filedes, int cmd, ...) {
  __rtsan_notify_intercepted_call("fcntl64");

  va_list args;
  va_start(args, cmd);

  // Following precedent here. The linux source (fcntl.c, do_fcntl) accepts the
  // final argument in a variable that will hold the largest of the possible
  // argument types (pointers and ints are typical in fcntl) It is then assumed
  // that the implementation of fcntl will cast it properly depending on cmd.
  //
  // This is also similar to what is done in
  // sanitizer_common/sanitizer_common_syscalls.inc
  const unsigned long arg = va_arg(args, unsigned long);
  int result = REAL(fcntl64)(filedes, cmd, arg);

  va_end(args);

  return result;
}
#define RTSAN_MAYBE_INTERCEPT_FCNTL64 INTERCEPT_FUNCTION(fcntl64)
#else
#define RTSAN_MAYBE_INTERCEPT_FCNTL64
#endif // SANITIZER_INTERCEPT_FCNTL64

INTERCEPTOR(int, close, int filedes) {
  __rtsan_notify_intercepted_call("close");
  return REAL(close)(filedes);
}

INTERCEPTOR(FILE *, fopen, const char *path, const char *mode) {
  __rtsan_notify_intercepted_call("fopen");
  return REAL(fopen)(path, mode);
}

INTERCEPTOR(FILE *, freopen, const char *path, const char *mode, FILE *stream) {
  __rtsan_notify_intercepted_call("freopen");
  return REAL(freopen)(path, mode, stream);
}

// Streams

#if SANITIZER_INTERCEPT_FOPEN64
INTERCEPTOR(FILE *, fopen64, const char *path, const char *mode) {
  __rtsan_notify_intercepted_call("fopen64");
  return REAL(fopen64)(path, mode);
}

INTERCEPTOR(FILE *, freopen64, const char *path, const char *mode,
            FILE *stream) {
  __rtsan_notify_intercepted_call("freopen64");
  return REAL(freopen64)(path, mode, stream);
}
#define RTSAN_MAYBE_INTERCEPT_FOPEN64 INTERCEPT_FUNCTION(fopen64);
#define RTSAN_MAYBE_INTERCEPT_FREOPEN64 INTERCEPT_FUNCTION(freopen64);
#else
#define RTSAN_MAYBE_INTERCEPT_FOPEN64
#define RTSAN_MAYBE_INTERCEPT_FREOPEN64
#endif // SANITIZER_INTERCEPT_FOPEN64

INTERCEPTOR(size_t, fread, void *ptr, size_t size, size_t nitems,
            FILE *stream) {
  __rtsan_notify_intercepted_call("fread");
  return REAL(fread)(ptr, size, nitems, stream);
}

INTERCEPTOR(size_t, fwrite, const void *ptr, size_t size, size_t nitems,
            FILE *stream) {
  __rtsan_notify_intercepted_call("fwrite");
  return REAL(fwrite)(ptr, size, nitems, stream);
}

INTERCEPTOR(int, fclose, FILE *stream) {
  __rtsan_notify_intercepted_call("fclose");
  return REAL(fclose)(stream);
}

INTERCEPTOR(int, fputs, const char *s, FILE *stream) {
  __rtsan_notify_intercepted_call("fputs");
  return REAL(fputs)(s, stream);
}

INTERCEPTOR(int, fflush, FILE *stream) {
  __rtsan_notify_intercepted_call("fflush");
  return REAL(fflush)(stream);
}

#if SANITIZER_APPLE
INTERCEPTOR(int, fpurge, FILE *stream) {
  __rtsan_notify_intercepted_call("fpurge");
  return REAL(fpurge)(stream);
}
#endif

INTERCEPTOR(FILE *, fdopen, int fd, const char *mode) {
  __rtsan_notify_intercepted_call("fdopen");
  return REAL(fdopen)(fd, mode);
}

#if SANITIZER_INTERCEPT_FOPENCOOKIE
INTERCEPTOR(FILE *, fopencookie, void *cookie, const char *mode,
            cookie_io_functions_t funcs) {
  __rtsan_notify_intercepted_call("fopencookie");
  return REAL(fopencookie)(cookie, mode, funcs);
}
#define RTSAN_MAYBE_INTERCEPT_FOPENCOOKIE INTERCEPT_FUNCTION(fopencookie)
#else
#define RTSAN_MAYBE_INTERCEPT_FOPENCOOKIE
#endif

#if SANITIZER_INTERCEPT_OPEN_MEMSTREAM
INTERCEPTOR(FILE *, open_memstream, char **buf, size_t *size) {
  __rtsan_notify_intercepted_call("open_memstream");
  return REAL(open_memstream)(buf, size);
}

INTERCEPTOR(FILE *, fmemopen, void *buf, size_t size, const char *mode) {
  __rtsan_notify_intercepted_call("fmemopen");
  return REAL(fmemopen)(buf, size, mode);
}
#define RTSAN_MAYBE_INTERCEPT_OPEN_MEMSTREAM INTERCEPT_FUNCTION(open_memstream)
#define RTSAN_MAYBE_INTERCEPT_FMEMOPEN INTERCEPT_FUNCTION(fmemopen)
#else
#define RTSAN_MAYBE_INTERCEPT_OPEN_MEMSTREAM
#define RTSAN_MAYBE_INTERCEPT_FMEMOPEN
#endif

#if SANITIZER_INTERCEPT_SETVBUF
INTERCEPTOR(void, setbuf, FILE *stream, char *buf) {
  __rtsan_notify_intercepted_call("setbuf");
  return REAL(setbuf)(stream, buf);
}

INTERCEPTOR(int, setvbuf, FILE *stream, char *buf, int mode, size_t size) {
  __rtsan_notify_intercepted_call("setvbuf");
  return REAL(setvbuf)(stream, buf, mode, size);
}

#if SANITIZER_LINUX
INTERCEPTOR(void, setlinebuf, FILE *stream) {
#else
INTERCEPTOR(int, setlinebuf, FILE *stream) {
#endif
  __rtsan_notify_intercepted_call("setlinebuf");
  return REAL(setlinebuf)(stream);
}

#if SANITIZER_LINUX
INTERCEPTOR(void, setbuffer, FILE *stream, char *buf, size_t size) {
#else
INTERCEPTOR(void, setbuffer, FILE *stream, char *buf, int size) {
#endif
  __rtsan_notify_intercepted_call("setbuffer");
  return REAL(setbuffer)(stream, buf, size);
}
#define RTSAN_MAYBE_INTERCEPT_SETBUF INTERCEPT_FUNCTION(setbuf)
#define RTSAN_MAYBE_INTERCEPT_SETVBUF INTERCEPT_FUNCTION(setvbuf)
#define RTSAN_MAYBE_INTERCEPT_SETLINEBUF INTERCEPT_FUNCTION(setlinebuf)
#define RTSAN_MAYBE_INTERCEPT_SETBUFFER INTERCEPT_FUNCTION(setbuffer)
#else
#define RTSAN_MAYBE_INTERCEPT_SETBUF
#define RTSAN_MAYBE_INTERCEPT_SETVBUF
#define RTSAN_MAYBE_INTERCEPT_SETLINEBUF
#define RTSAN_MAYBE_INTERCEPT_SETBUFFER
#endif

INTERCEPTOR(int, puts, const char *s) {
  __rtsan_notify_intercepted_call("puts");
  return REAL(puts)(s);
}

INTERCEPTOR(ssize_t, read, int fd, void *buf, size_t count) {
  __rtsan_notify_intercepted_call("read");
  return REAL(read)(fd, buf, count);
}

INTERCEPTOR(ssize_t, write, int fd, const void *buf, size_t count) {
  __rtsan_notify_intercepted_call("write");
  return REAL(write)(fd, buf, count);
}

INTERCEPTOR(ssize_t, pread, int fd, void *buf, size_t count, off_t offset) {
  __rtsan_notify_intercepted_call("pread");
  return REAL(pread)(fd, buf, count, offset);
}

#if SANITIZER_INTERCEPT_PREAD64
INTERCEPTOR(ssize_t, pread64, int fd, void *buf, size_t count, off_t offset) {
  __rtsan_notify_intercepted_call("pread64");
  return REAL(pread64)(fd, buf, count, offset);
}
#define RTSAN_MAYBE_INTERCEPT_PREAD64 INTERCEPT_FUNCTION(pread64)
#else
#define RTSAN_MAYBE_INTERCEPT_PREAD64
#endif // SANITIZER_INTERCEPT_PREAD64

INTERCEPTOR(ssize_t, readv, int fd, const struct iovec *iov, int iovcnt) {
  __rtsan_notify_intercepted_call("readv");
  return REAL(readv)(fd, iov, iovcnt);
}

INTERCEPTOR(ssize_t, pwrite, int fd, const void *buf, size_t count,
            off_t offset) {
  __rtsan_notify_intercepted_call("pwrite");
  return REAL(pwrite)(fd, buf, count, offset);
}

#if SANITIZER_INTERCEPT_PWRITE64
INTERCEPTOR(ssize_t, pwrite64, int fd, const void *buf, size_t count,
            off_t offset) {
  __rtsan_notify_intercepted_call("pwrite64");
  return REAL(pwrite64)(fd, buf, count, offset);
}
#define RTSAN_MAYBE_INTERCEPT_PWRITE64 INTERCEPT_FUNCTION(pwrite64)
#else
#define RTSAN_MAYBE_INTERCEPT_PWRITE64
#endif // SANITIZER_INTERCEPT_PWRITE64

INTERCEPTOR(ssize_t, writev, int fd, const struct iovec *iov, int iovcnt) {
  __rtsan_notify_intercepted_call("writev");
  return REAL(writev)(fd, iov, iovcnt);
}

INTERCEPTOR(off_t, lseek, int fd, off_t offset, int whence) {
  __rtsan_notify_intercepted_call("lseek");
  return REAL(lseek)(fd, offset, whence);
}

#if SANITIZER_INTERCEPT_LSEEK64
INTERCEPTOR(off64_t, lseek64, int fd, off64_t offset, int whence) {
  __rtsan_notify_intercepted_call("lseek64");
  return REAL(lseek64)(fd, offset, whence);
}
#define RTSAN_MAYBE_INTERCEPT_LSEEK64 INTERCEPT_FUNCTION(lseek64)
#else
#define RTSAN_MAYBE_INTERCEPT_LSEEK64
#endif // SANITIZER_INTERCEPT_LSEEK64

INTERCEPTOR(int, dup, int oldfd) {
  __rtsan_notify_intercepted_call("dup");
  return REAL(dup)(oldfd);
}

INTERCEPTOR(int, dup2, int oldfd, int newfd) {
  __rtsan_notify_intercepted_call("dup2");
  return REAL(dup2)(oldfd, newfd);
}

INTERCEPTOR(int, chmod, const char *path, mode_t mode) {
  __rtsan_notify_intercepted_call("chmod");
  return REAL(chmod)(path, mode);
}

INTERCEPTOR(int, fchmod, int fd, mode_t mode) {
  __rtsan_notify_intercepted_call("fchmod");
  return REAL(fchmod)(fd, mode);
}

INTERCEPTOR(int, mkdir, const char *path, mode_t mode) {
  __rtsan_notify_intercepted_call("mkdir");
  return REAL(mkdir)(path, mode);
}

INTERCEPTOR(int, rmdir, const char *path) {
  __rtsan_notify_intercepted_call("rmdir");
  return REAL(rmdir)(path);
}

INTERCEPTOR(mode_t, umask, mode_t cmask) {
  __rtsan_notify_intercepted_call("umask");
  return REAL(umask)(cmask);
}

// Concurrency
#if SANITIZER_APPLE
#pragma clang diagnostic push
// OSSpinLockLock is deprecated, but still in use in libc++
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
INTERCEPTOR(void, OSSpinLockLock, volatile OSSpinLock *lock) {
  __rtsan_notify_intercepted_call("OSSpinLockLock");
  return REAL(OSSpinLockLock)(lock);
}
#pragma clang diagnostic pop
#define RTSAN_MAYBE_INTERCEPT_OSSPINLOCKLOCK INTERCEPT_FUNCTION(OSSpinLockLock)
#else
#define RTSAN_MAYBE_INTERCEPT_OSSPINLOCKLOCK
#endif // SANITIZER_APPLE

#if SANITIZER_APPLE
INTERCEPTOR(void, os_unfair_lock_lock, os_unfair_lock_t lock) {
  __rtsan_notify_intercepted_call("os_unfair_lock_lock");
  return REAL(os_unfair_lock_lock)(lock);
}
#define RTSAN_MAYBE_INTERCEPT_OS_UNFAIR_LOCK_LOCK                              \
  INTERCEPT_FUNCTION(os_unfair_lock_lock)
#else
#define RTSAN_MAYBE_INTERCEPT_OS_UNFAIR_LOCK_LOCK
#endif // SANITIZER_APPLE

#if SANITIZER_LINUX
INTERCEPTOR(int, pthread_spin_lock, pthread_spinlock_t *spinlock) {
  __rtsan_notify_intercepted_call("pthread_spin_lock");
  return REAL(pthread_spin_lock)(spinlock);
}
#define RTSAN_MAYBE_INTERCEPT_PTHREAD_SPIN_LOCK                                \
  INTERCEPT_FUNCTION(pthread_spin_lock)
#else
#define RTSAN_MAYBE_INTERCEPT_PTHREAD_SPIN_LOCK
#endif // SANITIZER_LINUX

INTERCEPTOR(int, pthread_create, pthread_t *thread, const pthread_attr_t *attr,
            void *(*start_routine)(void *), void *arg) {
  __rtsan_notify_intercepted_call("pthread_create");
  return REAL(pthread_create)(thread, attr, start_routine, arg);
}

INTERCEPTOR(int, pthread_mutex_lock, pthread_mutex_t *mutex) {
  __rtsan_notify_intercepted_call("pthread_mutex_lock");
  return REAL(pthread_mutex_lock)(mutex);
}

INTERCEPTOR(int, pthread_mutex_unlock, pthread_mutex_t *mutex) {
  __rtsan_notify_intercepted_call("pthread_mutex_unlock");
  return REAL(pthread_mutex_unlock)(mutex);
}

INTERCEPTOR(int, pthread_join, pthread_t thread, void **value_ptr) {
  __rtsan_notify_intercepted_call("pthread_join");
  return REAL(pthread_join)(thread, value_ptr);
}

INTERCEPTOR(int, pthread_cond_signal, pthread_cond_t *cond) {
  __rtsan_notify_intercepted_call("pthread_cond_signal");
  return REAL(pthread_cond_signal)(cond);
}

INTERCEPTOR(int, pthread_cond_broadcast, pthread_cond_t *cond) {
  __rtsan_notify_intercepted_call("pthread_cond_broadcast");
  return REAL(pthread_cond_broadcast)(cond);
}

INTERCEPTOR(int, pthread_cond_wait, pthread_cond_t *cond,
            pthread_mutex_t *mutex) {
  __rtsan_notify_intercepted_call("pthread_cond_wait");
  return REAL(pthread_cond_wait)(cond, mutex);
}

INTERCEPTOR(int, pthread_cond_timedwait, pthread_cond_t *cond,
            pthread_mutex_t *mutex, const timespec *ts) {
  __rtsan_notify_intercepted_call("pthread_cond_timedwait");
  return REAL(pthread_cond_timedwait)(cond, mutex, ts);
}

INTERCEPTOR(int, pthread_rwlock_rdlock, pthread_rwlock_t *lock) {
  __rtsan_notify_intercepted_call("pthread_rwlock_rdlock");
  return REAL(pthread_rwlock_rdlock)(lock);
}

INTERCEPTOR(int, pthread_rwlock_unlock, pthread_rwlock_t *lock) {
  __rtsan_notify_intercepted_call("pthread_rwlock_unlock");
  return REAL(pthread_rwlock_unlock)(lock);
}

INTERCEPTOR(int, pthread_rwlock_wrlock, pthread_rwlock_t *lock) {
  __rtsan_notify_intercepted_call("pthread_rwlock_wrlock");
  return REAL(pthread_rwlock_wrlock)(lock);
}

// Sleeping

INTERCEPTOR(unsigned int, sleep, unsigned int s) {
  __rtsan_notify_intercepted_call("sleep");
  return REAL(sleep)(s);
}

INTERCEPTOR(int, usleep, useconds_t u) {
  __rtsan_notify_intercepted_call("usleep");
  return REAL(usleep)(u);
}

INTERCEPTOR(int, nanosleep, const struct timespec *rqtp,
            struct timespec *rmtp) {
  __rtsan_notify_intercepted_call("nanosleep");
  return REAL(nanosleep)(rqtp, rmtp);
}

INTERCEPTOR(int, sched_yield, void) {
  __rtsan_notify_intercepted_call("sched_yield");
  return REAL(sched_yield)();
}

// Memory

INTERCEPTOR(void *, calloc, SIZE_T num, SIZE_T size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Callocate(num, size);

  __rtsan_notify_intercepted_call("calloc");
  return REAL(calloc)(num, size);
}

INTERCEPTOR(void, free, void *ptr) {
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);

  // According to the C and C++ standard, freeing a nullptr is guaranteed to be
  // a no-op (and thus real-time safe). This can be confirmed for looking at
  // __libc_free in the glibc source.
  if (ptr != nullptr)
    __rtsan_notify_intercepted_call("free");

  return REAL(free)(ptr);
}

INTERCEPTOR(void *, malloc, SIZE_T size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Allocate(size);

  __rtsan_notify_intercepted_call("malloc");
  return REAL(malloc)(size);
}

INTERCEPTOR(void *, realloc, void *ptr, SIZE_T size) {
  if (DlsymAlloc::Use() || DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Realloc(ptr, size);

  __rtsan_notify_intercepted_call("realloc");
  return REAL(realloc)(ptr, size);
}

INTERCEPTOR(void *, reallocf, void *ptr, SIZE_T size) {
  __rtsan_notify_intercepted_call("reallocf");
  return REAL(reallocf)(ptr, size);
}

INTERCEPTOR(void *, valloc, SIZE_T size) {
  __rtsan_notify_intercepted_call("valloc");
  return REAL(valloc)(size);
}

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC

// In some cases, when targeting older Darwin versions, this warning may pop up.
// Because we are providing a wrapper, the client is responsible to check
// whether aligned_alloc is available, not us. We still succeed linking on an
// old OS, because we are using a weak symbol (see aligned_alloc in
// sanitizer_platform_interceptors.h)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
INTERCEPTOR(void *, aligned_alloc, SIZE_T alignment, SIZE_T size) {
  __rtsan_notify_intercepted_call("aligned_alloc");
  return REAL(aligned_alloc)(alignment, size);
}
#pragma clang diagnostic pop
#define RTSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC INTERCEPT_FUNCTION(aligned_alloc)
#else
#define RTSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC
#endif

INTERCEPTOR(int, posix_memalign, void **memptr, size_t alignment, size_t size) {
  __rtsan_notify_intercepted_call("posix_memalign");
  return REAL(posix_memalign)(memptr, alignment, size);
}

#if SANITIZER_INTERCEPT_MEMALIGN
INTERCEPTOR(void *, memalign, size_t alignment, size_t size) {
  __rtsan_notify_intercepted_call("memalign");
  return REAL(memalign)(alignment, size);
}
#define RTSAN_MAYBE_INTERCEPT_MEMALIGN INTERCEPT_FUNCTION(memalign)
#else
#define RTSAN_MAYBE_INTERCEPT_MEMALIGN
#endif

#if SANITIZER_INTERCEPT_PVALLOC
INTERCEPTOR(void *, pvalloc, size_t size) {
  __rtsan_notify_intercepted_call("pvalloc");
  return REAL(pvalloc)(size);
}
#define RTSAN_MAYBE_INTERCEPT_PVALLOC INTERCEPT_FUNCTION(pvalloc)
#else
#define RTSAN_MAYBE_INTERCEPT_PVALLOC
#endif

INTERCEPTOR(void *, mmap, void *addr, size_t length, int prot, int flags,
            int fd, off_t offset) {
  __rtsan_notify_intercepted_call("mmap");
  return REAL(mmap)(addr, length, prot, flags, fd, offset);
}

#if SANITIZER_INTERCEPT_MMAP64
INTERCEPTOR(void *, mmap64, void *addr, size_t length, int prot, int flags,
            int fd, off64_t offset) {
  __rtsan_notify_intercepted_call("mmap64");
  return REAL(mmap64)(addr, length, prot, flags, fd, offset);
}
#define RTSAN_MAYBE_INTERCEPT_MMAP64 INTERCEPT_FUNCTION(mmap64)
#else
#define RTSAN_MAYBE_INTERCEPT_MMAP64
#endif // SANITIZER_INTERCEPT_MMAP64

INTERCEPTOR(int, munmap, void *addr, size_t length) {
  __rtsan_notify_intercepted_call("munmap");
  return REAL(munmap)(addr, length);
}

INTERCEPTOR(int, shm_open, const char *name, int oflag, mode_t mode) {
  __rtsan_notify_intercepted_call("shm_open");
  return REAL(shm_open)(name, oflag, mode);
}

INTERCEPTOR(int, shm_unlink, const char *name) {
  __rtsan_notify_intercepted_call("shm_unlink");
  return REAL(shm_unlink)(name);
}

// Sockets
INTERCEPTOR(int, getaddrinfo, const char *node, const char *service,
            const struct addrinfo *hints, struct addrinfo **res) {
  __rtsan_notify_intercepted_call("getaddrinfo");
  return REAL(getaddrinfo)(node, service, hints, res);
}

INTERCEPTOR(int, getnameinfo, const struct sockaddr *sa, socklen_t salen,
            char *host, socklen_t hostlen, char *serv, socklen_t servlen,
            int flags) {
  __rtsan_notify_intercepted_call("getnameinfo");
  return REAL(getnameinfo)(sa, salen, host, hostlen, serv, servlen, flags);
}

INTERCEPTOR(int, bind, int socket, const struct sockaddr *address,
            socklen_t address_len) {
  __rtsan_notify_intercepted_call("bind");
  return REAL(bind)(socket, address, address_len);
}

INTERCEPTOR(int, listen, int socket, int backlog) {
  __rtsan_notify_intercepted_call("listen");
  return REAL(listen)(socket, backlog);
}

INTERCEPTOR(int, accept, int socket, struct sockaddr *address,
            socklen_t *address_len) {
  __rtsan_notify_intercepted_call("accept");
  return REAL(accept)(socket, address, address_len);
}

INTERCEPTOR(int, connect, int socket, const struct sockaddr *address,
            socklen_t address_len) {
  __rtsan_notify_intercepted_call("connect");
  return REAL(connect)(socket, address, address_len);
}

INTERCEPTOR(int, socket, int domain, int type, int protocol) {
  __rtsan_notify_intercepted_call("socket");
  return REAL(socket)(domain, type, protocol);
}

INTERCEPTOR(ssize_t, send, int sockfd, const void *buf, size_t len, int flags) {
  __rtsan_notify_intercepted_call("send");
  return REAL(send)(sockfd, buf, len, flags);
}

INTERCEPTOR(ssize_t, sendmsg, int socket, const struct msghdr *message,
            int flags) {
  __rtsan_notify_intercepted_call("sendmsg");
  return REAL(sendmsg)(socket, message, flags);
}

INTERCEPTOR(ssize_t, sendto, int socket, const void *buffer, size_t length,
            int flags, const struct sockaddr *dest_addr, socklen_t dest_len) {
  __rtsan_notify_intercepted_call("sendto");
  return REAL(sendto)(socket, buffer, length, flags, dest_addr, dest_len);
}

INTERCEPTOR(ssize_t, recv, int socket, void *buffer, size_t length, int flags) {
  __rtsan_notify_intercepted_call("recv");
  return REAL(recv)(socket, buffer, length, flags);
}

INTERCEPTOR(ssize_t, recvfrom, int socket, void *buffer, size_t length,
            int flags, struct sockaddr *address, socklen_t *address_len) {
  __rtsan_notify_intercepted_call("recvfrom");
  return REAL(recvfrom)(socket, buffer, length, flags, address, address_len);
}

INTERCEPTOR(ssize_t, recvmsg, int socket, struct msghdr *message, int flags) {
  __rtsan_notify_intercepted_call("recvmsg");
  return REAL(recvmsg)(socket, message, flags);
}

INTERCEPTOR(int, shutdown, int socket, int how) {
  __rtsan_notify_intercepted_call("shutdown");
  return REAL(shutdown)(socket, how);
}

#if SANITIZER_INTERCEPT_ACCEPT4
INTERCEPTOR(int, accept4, int socket, struct sockaddr *address,
            socklen_t *address_len, int flags) {
  __rtsan_notify_intercepted_call("accept4");
  return REAL(accept4)(socket, address, address_len, flags);
}
#define RTSAN_MAYBE_INTERCEPT_ACCEPT4 INTERCEPT_FUNCTION(accept4)
#else
#define RTSAN_MAYBE_INTERCEPT_ACCEPT4
#endif

// I/O Multiplexing

INTERCEPTOR(int, poll, struct pollfd *fds, nfds_t nfds, int timeout) {
  __rtsan_notify_intercepted_call("poll");
  return REAL(poll)(fds, nfds, timeout);
}

#if !SANITIZER_APPLE
// FIXME: This should work on all unix systems, even Mac, but currently
// it is showing some weird error while linking
// error: declaration of 'select' has a different language linkage
INTERCEPTOR(int, select, int nfds, fd_set *readfds, fd_set *writefds,
            fd_set *exceptfds, struct timeval *timeout) {
  __rtsan_notify_intercepted_call("select");
  return REAL(select)(nfds, readfds, writefds, exceptfds, timeout);
}
#define RTSAN_MAYBE_INTERCEPT_SELECT INTERCEPT_FUNCTION(select)
#else
#define RTSAN_MAYBE_INTERCEPT_SELECT
#endif // !SANITIZER_APPLE

INTERCEPTOR(int, pselect, int nfds, fd_set *readfds, fd_set *writefds,
            fd_set *exceptfds, const struct timespec *timeout,
            const sigset_t *sigmask) {
  __rtsan_notify_intercepted_call("pselect");
  return REAL(pselect)(nfds, readfds, writefds, exceptfds, timeout, sigmask);
}

#if SANITIZER_INTERCEPT_EPOLL
INTERCEPTOR(int, epoll_create, int size) {
  __rtsan_notify_intercepted_call("epoll_create");
  return REAL(epoll_create)(size);
}

INTERCEPTOR(int, epoll_create1, int flags) {
  __rtsan_notify_intercepted_call("epoll_create1");
  return REAL(epoll_create1)(flags);
}

INTERCEPTOR(int, epoll_ctl, int epfd, int op, int fd,
            struct epoll_event *event) {
  __rtsan_notify_intercepted_call("epoll_ctl");
  return REAL(epoll_ctl)(epfd, op, fd, event);
}

INTERCEPTOR(int, epoll_wait, int epfd, struct epoll_event *events,
            int maxevents, int timeout) {
  __rtsan_notify_intercepted_call("epoll_wait");
  return REAL(epoll_wait)(epfd, events, maxevents, timeout);
}

INTERCEPTOR(int, epoll_pwait, int epfd, struct epoll_event *events,
            int maxevents, int timeout, const sigset_t *sigmask) {
  __rtsan_notify_intercepted_call("epoll_pwait");
  return REAL(epoll_pwait)(epfd, events, maxevents, timeout, sigmask);
}
#define RTSAN_MAYBE_INTERCEPT_EPOLL_CREATE INTERCEPT_FUNCTION(epoll_create)
#define RTSAN_MAYBE_INTERCEPT_EPOLL_CREATE1 INTERCEPT_FUNCTION(epoll_create1)
#define RTSAN_MAYBE_INTERCEPT_EPOLL_CTL INTERCEPT_FUNCTION(epoll_ctl)
#define RTSAN_MAYBE_INTERCEPT_EPOLL_WAIT INTERCEPT_FUNCTION(epoll_wait)
#define RTSAN_MAYBE_INTERCEPT_EPOLL_PWAIT INTERCEPT_FUNCTION(epoll_pwait)
#else
#define RTSAN_MAYBE_INTERCEPT_EPOLL_CREATE
#define RTSAN_MAYBE_INTERCEPT_EPOLL_CREATE1
#define RTSAN_MAYBE_INTERCEPT_EPOLL_CTL
#define RTSAN_MAYBE_INTERCEPT_EPOLL_WAIT
#define RTSAN_MAYBE_INTERCEPT_EPOLL_PWAIT
#endif // SANITIZER_INTERCEPT_EPOLL

#if SANITIZER_INTERCEPT_PPOLL
INTERCEPTOR(int, ppoll, struct pollfd *fds, nfds_t n, const struct timespec *ts,
            const sigset_t *set) {
  __rtsan_notify_intercepted_call("ppoll");
  return REAL(ppoll)(fds, n, ts, set);
}
#define RTSAN_MAYBE_INTERCEPT_PPOLL INTERCEPT_FUNCTION(ppoll)
#else
#define RTSAN_MAYBE_INTERCEPT_PPOLL
#endif

#if SANITIZER_INTERCEPT_KQUEUE
INTERCEPTOR(int, kqueue, void) {
  __rtsan_notify_intercepted_call("kqueue");
  return REAL(kqueue)();
}

INTERCEPTOR(int, kevent, int kq, const struct kevent *changelist, int nchanges,
            struct kevent *eventlist, int nevents,
            const struct timespec *timeout) {
  __rtsan_notify_intercepted_call("kevent");
  return REAL(kevent)(kq, changelist, nchanges, eventlist, nevents, timeout);
}

INTERCEPTOR(int, kevent64, int kq, const struct kevent64_s *changelist,
            int nchanges, struct kevent64_s *eventlist, int nevents,
            unsigned int flags, const struct timespec *timeout) {
  __rtsan_notify_intercepted_call("kevent64");
  return REAL(kevent64)(kq, changelist, nchanges, eventlist, nevents, flags,
                        timeout);
}
#define RTSAN_MAYBE_INTERCEPT_KQUEUE INTERCEPT_FUNCTION(kqueue)
#define RTSAN_MAYBE_INTERCEPT_KEVENT INTERCEPT_FUNCTION(kevent)
#define RTSAN_MAYBE_INTERCEPT_KEVENT64 INTERCEPT_FUNCTION(kevent64)
#else
#define RTSAN_MAYBE_INTERCEPT_KQUEUE
#define RTSAN_MAYBE_INTERCEPT_KEVENT
#define RTSAN_MAYBE_INTERCEPT_KEVENT64
#endif // SANITIZER_INTERCEPT_KQUEUE

INTERCEPTOR(int, pipe, int pipefd[2]) {
  __rtsan_notify_intercepted_call("pipe");
  return REAL(pipe)(pipefd);
}

INTERCEPTOR(int, mkfifo, const char *pathname, mode_t mode) {
  __rtsan_notify_intercepted_call("mkfifo");
  return REAL(mkfifo)(pathname, mode);
}

INTERCEPTOR(pid_t, fork, void) {
  __rtsan_notify_intercepted_call("fork");
  return REAL(fork)();
}

INTERCEPTOR(int, execve, const char *filename, char *const argv[],
            char *const envp[]) {
  __rtsan_notify_intercepted_call("execve");
  return REAL(execve)(filename, argv, envp);
}

// TODO: the `wait` family of functions is an oddity. In testing, if you
// intercept them, Darwin seemingly ignores them, and linux never returns from
// the test. Revisit this in the future, but hopefully intercepting fork/exec is
// enough to dissuade usage of wait by proxy.

#if SANITIZER_APPLE
#define INT_TYPE_SYSCALL int
#else
#define INT_TYPE_SYSCALL long
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
INTERCEPTOR(INT_TYPE_SYSCALL, syscall, INT_TYPE_SYSCALL number, ...) {
  __rtsan_notify_intercepted_call("syscall");

  va_list args;
  va_start(args, number);

  // the goal is to pick something large enough to hold all syscall args
  // see fcntl for more discussion and why we always pull all 6 args
  using arg_type = unsigned long;
  arg_type arg1 = va_arg(args, arg_type);
  arg_type arg2 = va_arg(args, arg_type);
  arg_type arg3 = va_arg(args, arg_type);
  arg_type arg4 = va_arg(args, arg_type);
  arg_type arg5 = va_arg(args, arg_type);
  arg_type arg6 = va_arg(args, arg_type);

  // these are various examples of things that COULD be passed
  static_assert(sizeof(arg_type) >= sizeof(off_t));
  static_assert(sizeof(arg_type) >= sizeof(struct flock *));
  static_assert(sizeof(arg_type) >= sizeof(const char *));
  static_assert(sizeof(arg_type) >= sizeof(int));
  static_assert(sizeof(arg_type) >= sizeof(unsigned long));

  va_end(args);

  return REAL(syscall)(number, arg1, arg2, arg3, arg4, arg5, arg6);
}
#pragma clang diagnostic pop

// Preinit
void __rtsan::InitializeInterceptors() {
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(free);
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(reallocf);
  INTERCEPT_FUNCTION(valloc);
  RTSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC;
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(mmap);
  RTSAN_MAYBE_INTERCEPT_MMAP64;
  INTERCEPT_FUNCTION(munmap);
  INTERCEPT_FUNCTION(shm_open);
  INTERCEPT_FUNCTION(shm_unlink);
  RTSAN_MAYBE_INTERCEPT_MEMALIGN;
  RTSAN_MAYBE_INTERCEPT_PVALLOC;

  INTERCEPT_FUNCTION(open);
  RTSAN_MAYBE_INTERCEPT_OPEN64;
  INTERCEPT_FUNCTION(openat);
  RTSAN_MAYBE_INTERCEPT_OPENAT64;
  INTERCEPT_FUNCTION(close);
  INTERCEPT_FUNCTION(fopen);
  RTSAN_MAYBE_INTERCEPT_FOPEN64;
  RTSAN_MAYBE_INTERCEPT_FREOPEN64;
  INTERCEPT_FUNCTION(fread);
  INTERCEPT_FUNCTION(read);
  INTERCEPT_FUNCTION(write);
  INTERCEPT_FUNCTION(pread);
  RTSAN_MAYBE_INTERCEPT_PREAD64;
  INTERCEPT_FUNCTION(readv);
  INTERCEPT_FUNCTION(pwrite);
  RTSAN_MAYBE_INTERCEPT_PWRITE64;
  INTERCEPT_FUNCTION(writev);
  INTERCEPT_FUNCTION(fwrite);
  INTERCEPT_FUNCTION(fclose);
  INTERCEPT_FUNCTION(fcntl);
  RTSAN_MAYBE_INTERCEPT_FCNTL64;
  INTERCEPT_FUNCTION(creat);
  RTSAN_MAYBE_INTERCEPT_CREAT64;
  INTERCEPT_FUNCTION(puts);
  INTERCEPT_FUNCTION(fputs);
  INTERCEPT_FUNCTION(fflush);
  INTERCEPT_FUNCTION(fdopen);
  INTERCEPT_FUNCTION(freopen);
  RTSAN_MAYBE_INTERCEPT_FOPENCOOKIE;
  RTSAN_MAYBE_INTERCEPT_OPEN_MEMSTREAM;
  RTSAN_MAYBE_INTERCEPT_FMEMOPEN;
  RTSAN_MAYBE_INTERCEPT_SETBUF;
  RTSAN_MAYBE_INTERCEPT_SETVBUF;
  RTSAN_MAYBE_INTERCEPT_SETLINEBUF;
  RTSAN_MAYBE_INTERCEPT_SETBUFFER;
  INTERCEPT_FUNCTION(lseek);
  RTSAN_MAYBE_INTERCEPT_LSEEK64;
  INTERCEPT_FUNCTION(dup);
  INTERCEPT_FUNCTION(dup2);
  INTERCEPT_FUNCTION(chmod);
  INTERCEPT_FUNCTION(fchmod);
  INTERCEPT_FUNCTION(mkdir);
  INTERCEPT_FUNCTION(rmdir);
  INTERCEPT_FUNCTION(umask);
  INTERCEPT_FUNCTION(ioctl);

  RTSAN_MAYBE_INTERCEPT_OSSPINLOCKLOCK;
  RTSAN_MAYBE_INTERCEPT_OS_UNFAIR_LOCK_LOCK;
  RTSAN_MAYBE_INTERCEPT_PTHREAD_SPIN_LOCK;

  INTERCEPT_FUNCTION(pthread_create);
  INTERCEPT_FUNCTION(pthread_mutex_lock);
  INTERCEPT_FUNCTION(pthread_mutex_unlock);
  INTERCEPT_FUNCTION(pthread_join);
  INTERCEPT_FUNCTION(pthread_cond_signal);
  INTERCEPT_FUNCTION(pthread_cond_broadcast);
  INTERCEPT_FUNCTION(pthread_cond_wait);
  INTERCEPT_FUNCTION(pthread_cond_timedwait);
  INTERCEPT_FUNCTION(pthread_rwlock_rdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_unlock);
  INTERCEPT_FUNCTION(pthread_rwlock_wrlock);

  INTERCEPT_FUNCTION(sleep);
  INTERCEPT_FUNCTION(usleep);
  INTERCEPT_FUNCTION(nanosleep);
  INTERCEPT_FUNCTION(sched_yield);

  INTERCEPT_FUNCTION(accept);
  INTERCEPT_FUNCTION(bind);
  INTERCEPT_FUNCTION(connect);
  INTERCEPT_FUNCTION(getaddrinfo);
  INTERCEPT_FUNCTION(getnameinfo);
  INTERCEPT_FUNCTION(listen);
  INTERCEPT_FUNCTION(recv);
  INTERCEPT_FUNCTION(recvfrom);
  INTERCEPT_FUNCTION(recvmsg);
  INTERCEPT_FUNCTION(send);
  INTERCEPT_FUNCTION(sendmsg);
  INTERCEPT_FUNCTION(sendto);
  INTERCEPT_FUNCTION(shutdown);
  INTERCEPT_FUNCTION(socket);
  RTSAN_MAYBE_INTERCEPT_ACCEPT4;

  RTSAN_MAYBE_INTERCEPT_SELECT;
  INTERCEPT_FUNCTION(pselect);
  INTERCEPT_FUNCTION(poll);
  RTSAN_MAYBE_INTERCEPT_EPOLL_CREATE;
  RTSAN_MAYBE_INTERCEPT_EPOLL_CREATE1;
  RTSAN_MAYBE_INTERCEPT_EPOLL_CTL;
  RTSAN_MAYBE_INTERCEPT_EPOLL_WAIT;
  RTSAN_MAYBE_INTERCEPT_EPOLL_PWAIT;
  RTSAN_MAYBE_INTERCEPT_PPOLL;
  RTSAN_MAYBE_INTERCEPT_KQUEUE;
  RTSAN_MAYBE_INTERCEPT_KEVENT;
  RTSAN_MAYBE_INTERCEPT_KEVENT64;

  INTERCEPT_FUNCTION(pipe);
  INTERCEPT_FUNCTION(mkfifo);

  INTERCEPT_FUNCTION(fork);
  INTERCEPT_FUNCTION(execve);

  INTERCEPT_FUNCTION(syscall);
}

#endif // SANITIZER_POSIX
