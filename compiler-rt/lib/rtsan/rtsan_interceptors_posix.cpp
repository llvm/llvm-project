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
#include "sanitizer_common/sanitizer_glibc_version.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"

#include "interception/interception.h"
#include "rtsan/rtsan.h"

#if SANITIZER_APPLE
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
#if SANITIZER_LINUX
#include <linux/mman.h>
#include <sys/inotify.h>
#endif
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

using namespace __sanitizer;

DECLARE_REAL_AND_INTERCEPTOR(void *, malloc, usize size)
DECLARE_REAL_AND_INTERCEPTOR(void, free, void *ptr)

namespace {
struct DlsymAlloc : public DlSymAllocator<DlsymAlloc> {
  static bool UseImpl() { return !__rtsan_is_initialized(); }
};
} // namespace

// See note in tsan as to why this is necessary
static pthread_cond_t *init_cond(pthread_cond_t *c, bool force = false) {
  if (!common_flags()->legacy_pthread_cond)
    return c;

  atomic_uintptr_t *p = (atomic_uintptr_t *)c;
  uptr cond = atomic_load(p, memory_order_acquire);
  if (!force && cond != 0)
    return (pthread_cond_t *)cond;
  void *newcond = WRAP(malloc)(sizeof(pthread_cond_t));
  internal_memset(newcond, 0, sizeof(pthread_cond_t));
  if (atomic_compare_exchange_strong(p, &cond, (uptr)newcond,
                                     memory_order_acq_rel))
    return (pthread_cond_t *)newcond;
  WRAP(free)(newcond);
  return (pthread_cond_t *)cond;
}

static void destroy_cond(pthread_cond_t *cond) {
  if (common_flags()->legacy_pthread_cond) {
    // Free our aux cond and zero the pointer to not leave dangling pointers.
    WRAP(free)(cond);
    atomic_store((atomic_uintptr_t *)cond, 0, memory_order_relaxed);
  }
}

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

#if SANITIZER_MUSL
INTERCEPTOR(int, ioctl, int filedes, int request, ...) {
#else
INTERCEPTOR(int, ioctl, int filedes, unsigned long request, ...) {
#endif
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

INTERCEPTOR(int, chdir, const char *path) {
  __rtsan_notify_intercepted_call("chdir");
  return REAL(chdir)(path);
}

INTERCEPTOR(int, fchdir, int fd) {
  __rtsan_notify_intercepted_call("fchdir");
  return REAL(fchdir)(fd);
}

#if SANITIZER_INTERCEPT_READLINK
INTERCEPTOR(ssize_t, readlink, const char *pathname, char *buf, size_t size) {
  __rtsan_notify_intercepted_call("readlink");
  return REAL(readlink)(pathname, buf, size);
}
#define RTSAN_MAYBE_INTERCEPT_READLINK INTERCEPT_FUNCTION(readlink)
#else
#define RTSAN_MAYBE_INTERCEPT_READLINK
#endif

#if SANITIZER_INTERCEPT_READLINKAT
INTERCEPTOR(ssize_t, readlinkat, int dirfd, const char *pathname, char *buf,
            size_t size) {
  __rtsan_notify_intercepted_call("readlinkat");
  return REAL(readlinkat)(dirfd, pathname, buf, size);
}
#define RTSAN_MAYBE_INTERCEPT_READLINKAT INTERCEPT_FUNCTION(readlinkat)
#else
#define RTSAN_MAYBE_INTERCEPT_READLINKAT
#endif

INTERCEPTOR(int, unlink, const char *pathname) {
  __rtsan_notify_intercepted_call("unlink");
  return REAL(unlink)(pathname);
}

INTERCEPTOR(int, unlinkat, int fd, const char *pathname, int flag) {
  __rtsan_notify_intercepted_call("unlinkat");
  return REAL(unlinkat)(fd, pathname, flag);
}

INTERCEPTOR(int, truncate, const char *pathname, off_t length) {
  __rtsan_notify_intercepted_call("truncate");
  return REAL(truncate)(pathname, length);
}

INTERCEPTOR(int, ftruncate, int fd, off_t length) {
  __rtsan_notify_intercepted_call("ftruncate");
  return REAL(ftruncate)(fd, length);
}

#if SANITIZER_LINUX && !SANITIZER_MUSL
INTERCEPTOR(int, truncate64, const char *pathname, off64_t length) {
  __rtsan_notify_intercepted_call("truncate64");
  return REAL(truncate64)(pathname, length);
}

INTERCEPTOR(int, ftruncate64, int fd, off64_t length) {
  __rtsan_notify_intercepted_call("ftruncate64");
  return REAL(ftruncate64)(fd, length);
}
#define RTSAN_MAYBE_INTERCEPT_TRUNCATE64 INTERCEPT_FUNCTION(truncate64)
#define RTSAN_MAYBE_INTERCEPT_FTRUNCATE64 INTERCEPT_FUNCTION(ftruncate64)
#else
#define RTSAN_MAYBE_INTERCEPT_TRUNCATE64
#define RTSAN_MAYBE_INTERCEPT_FTRUNCATE64
#endif

INTERCEPTOR(int, symlink, const char *target, const char *linkpath) {
  __rtsan_notify_intercepted_call("symlink");
  return REAL(symlink)(target, linkpath);
}

INTERCEPTOR(int, symlinkat, const char *target, int newdirfd,
            const char *linkpath) {
  __rtsan_notify_intercepted_call("symlinkat");
  return REAL(symlinkat)(target, newdirfd, linkpath);
}

// Streams

INTERCEPTOR(FILE *, fopen, const char *path, const char *mode) {
  __rtsan_notify_intercepted_call("fopen");
  return REAL(fopen)(path, mode);
}

INTERCEPTOR(FILE *, freopen, const char *path, const char *mode, FILE *stream) {
  __rtsan_notify_intercepted_call("freopen");
  return REAL(freopen)(path, mode, stream);
}

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
#define RTSAN_MAYBE_INTERCEPT_FPURGE INTERCEPT_FUNCTION(fpurge)
#else
#define RTSAN_MAYBE_INTERCEPT_FPURGE
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

#if SANITIZER_INTERCEPT_FSEEK
INTERCEPTOR(int, fgetpos, FILE *stream, fpos_t *pos) {
  __rtsan_notify_intercepted_call("fgetpos");
  return REAL(fgetpos)(stream, pos);
}

INTERCEPTOR(int, fseek, FILE *stream, long offset, int whence) {
  __rtsan_notify_intercepted_call("fseek");
  return REAL(fseek)(stream, offset, whence);
}

INTERCEPTOR(int, fseeko, FILE *stream, off_t offset, int whence) {
  __rtsan_notify_intercepted_call("fseeko");
  return REAL(fseeko)(stream, offset, whence);
}

INTERCEPTOR(int, fsetpos, FILE *stream, const fpos_t *pos) {
  __rtsan_notify_intercepted_call("fsetpos");
  return REAL(fsetpos)(stream, pos);
}

INTERCEPTOR(long, ftell, FILE *stream) {
  __rtsan_notify_intercepted_call("ftell");
  return REAL(ftell)(stream);
}

INTERCEPTOR(off_t, ftello, FILE *stream) {
  __rtsan_notify_intercepted_call("ftello");
  return REAL(ftello)(stream);
}

#if SANITIZER_LINUX && !SANITIZER_MUSL
INTERCEPTOR(int, fgetpos64, FILE *stream, fpos64_t *pos) {
  __rtsan_notify_intercepted_call("fgetpos64");
  return REAL(fgetpos64)(stream, pos);
}

INTERCEPTOR(int, fseeko64, FILE *stream, off64_t offset, int whence) {
  __rtsan_notify_intercepted_call("fseeko64");
  return REAL(fseeko64)(stream, offset, whence);
}

INTERCEPTOR(int, fsetpos64, FILE *stream, const fpos64_t *pos) {
  __rtsan_notify_intercepted_call("fsetpos64");
  return REAL(fsetpos64)(stream, pos);
}

INTERCEPTOR(off64_t, ftello64, FILE *stream) {
  __rtsan_notify_intercepted_call("ftello64");
  return REAL(ftello64)(stream);
}
#endif

INTERCEPTOR(void, rewind, FILE *stream) {
  __rtsan_notify_intercepted_call("rewind");
  return REAL(rewind)(stream);
}
#define RTSAN_MAYBE_INTERCEPT_FGETPOS INTERCEPT_FUNCTION(fgetpos)
#define RTSAN_MAYBE_INTERCEPT_FSEEK INTERCEPT_FUNCTION(fseek)
#define RTSAN_MAYBE_INTERCEPT_FSEEKO INTERCEPT_FUNCTION(fseeko)
#define RTSAN_MAYBE_INTERCEPT_FSETPOS INTERCEPT_FUNCTION(fsetpos)
#define RTSAN_MAYBE_INTERCEPT_FTELL INTERCEPT_FUNCTION(ftell)
#define RTSAN_MAYBE_INTERCEPT_FTELLO INTERCEPT_FUNCTION(ftello)
#define RTSAN_MAYBE_INTERCEPT_REWIND INTERCEPT_FUNCTION(rewind)
#if SANITIZER_LINUX && !SANITIZER_MUSL
#define RTSAN_MAYBE_INTERCEPT_FGETPOS64 INTERCEPT_FUNCTION(fgetpos64)
#define RTSAN_MAYBE_INTERCEPT_FSEEKO64 INTERCEPT_FUNCTION(fseeko64)
#define RTSAN_MAYBE_INTERCEPT_FSETPOS64 INTERCEPT_FUNCTION(fsetpos64)
#define RTSAN_MAYBE_INTERCEPT_FTELLO64 INTERCEPT_FUNCTION(ftello64)
#else
#define RTSAN_MAYBE_INTERCEPT_FGETPOS64
#define RTSAN_MAYBE_INTERCEPT_FSEEKO64
#define RTSAN_MAYBE_INTERCEPT_FSETPOS64
#define RTSAN_MAYBE_INTERCEPT_FTELLO64
#endif
#else
#define RTSAN_MAYBE_INTERCEPT_FGETPOS
#define RTSAN_MAYBE_INTERCEPT_FSEEK
#define RTSAN_MAYBE_INTERCEPT_FSEEKO
#define RTSAN_MAYBE_INTERCEPT_FSETPOS
#define RTSAN_MAYBE_INTERCEPT_FTELL
#define RTSAN_MAYBE_INTERCEPT_FTELLO
#define RTSAN_MAYBE_INTERCEPT_REWIND
#define RTSAN_MAYBE_INTERCEPT_FGETPOS64
#define RTSAN_MAYBE_INTERCEPT_FSEEKO64
#define RTSAN_MAYBE_INTERCEPT_FSETPOS64
#define RTSAN_MAYBE_INTERCEPT_FTELLO64
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

#if SANITIZER_INTERCEPT_PREADV
INTERCEPTOR(ssize_t, preadv, int fd, const struct iovec *iov, int count,
            off_t offset) {
  __rtsan_notify_intercepted_call("preadv");
  return REAL(preadv)(fd, iov, count, offset);
}
#define RTSAN_MAYBE_INTERCEPT_PREADV INTERCEPT_FUNCTION(preadv)
#else
#define RTSAN_MAYBE_INTERCEPT_PREADV
#endif

#if SANITIZER_INTERCEPT_PREADV64
INTERCEPTOR(ssize_t, preadv64, int fd, const struct iovec *iov, int count,
            off_t offset) {
  __rtsan_notify_intercepted_call("preadv64");
  return REAL(preadv)(fd, iov, count, offset);
}
#define RTSAN_MAYBE_INTERCEPT_PREADV64 INTERCEPT_FUNCTION(preadv64)
#else
#define RTSAN_MAYBE_INTERCEPT_PREADV64
#endif

#if SANITIZER_INTERCEPT_PWRITEV
INTERCEPTOR(ssize_t, pwritev, int fd, const struct iovec *iov, int count,
            off_t offset) {
  __rtsan_notify_intercepted_call("pwritev");
  return REAL(pwritev)(fd, iov, count, offset);
}
#define RTSAN_MAYBE_INTERCEPT_PWRITEV INTERCEPT_FUNCTION(pwritev)
#else
#define RTSAN_MAYBE_INTERCEPT_PWRITEV
#endif

#if SANITIZER_INTERCEPT_PWRITEV64
INTERCEPTOR(ssize_t, pwritev64, int fd, const struct iovec *iov, int count,
            off_t offset) {
  __rtsan_notify_intercepted_call("pwritev64");
  return REAL(pwritev64)(fd, iov, count, offset);
}
#define RTSAN_MAYBE_INTERCEPT_PWRITEV64 INTERCEPT_FUNCTION(pwritev64)
#else
#define RTSAN_MAYBE_INTERCEPT_PWRITEV64
#endif

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
#undef OSSpinLockLock

INTERCEPTOR(void, OSSpinLockLock, volatile OSSpinLock *lock) {
  __rtsan_notify_intercepted_call("OSSpinLockLock");
  return REAL(OSSpinLockLock)(lock);
}

#define RTSAN_MAYBE_INTERCEPT_OSSPINLOCKLOCK INTERCEPT_FUNCTION(OSSpinLockLock)
#else
#define RTSAN_MAYBE_INTERCEPT_OSSPINLOCKLOCK
#endif // SANITIZER_APPLE

#if SANITIZER_APPLE
// _os_nospin_lock_lock may replace OSSpinLockLock due to deprecation macro.
typedef volatile OSSpinLock *_os_nospin_lock_t;

INTERCEPTOR(void, _os_nospin_lock_lock, _os_nospin_lock_t lock) {
  __rtsan_notify_intercepted_call("_os_nospin_lock_lock");
  return REAL(_os_nospin_lock_lock)(lock);
}
#pragma clang diagnostic pop // "-Wdeprecated-declarations"
#endif                       // SANITIZER_APPLE

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

INTERCEPTOR(int, pthread_cond_init, pthread_cond_t *cond,
            const pthread_condattr_t *a) {
  __rtsan_notify_intercepted_call("pthread_cond_init");
  pthread_cond_t *c = init_cond(cond, true);
  return REAL(pthread_cond_init)(c, a);
}

INTERCEPTOR(int, pthread_cond_signal, pthread_cond_t *cond) {
  __rtsan_notify_intercepted_call("pthread_cond_signal");
  pthread_cond_t *c = init_cond(cond);
  return REAL(pthread_cond_signal)(c);
}

INTERCEPTOR(int, pthread_cond_broadcast, pthread_cond_t *cond) {
  __rtsan_notify_intercepted_call("pthread_cond_broadcast");
  pthread_cond_t *c = init_cond(cond);
  return REAL(pthread_cond_broadcast)(c);
}

INTERCEPTOR(int, pthread_cond_wait, pthread_cond_t *cond,
            pthread_mutex_t *mutex) {
  __rtsan_notify_intercepted_call("pthread_cond_wait");
  pthread_cond_t *c = init_cond(cond);
  return REAL(pthread_cond_wait)(c, mutex);
}

INTERCEPTOR(int, pthread_cond_timedwait, pthread_cond_t *cond,
            pthread_mutex_t *mutex, const timespec *ts) {
  __rtsan_notify_intercepted_call("pthread_cond_timedwait");
  pthread_cond_t *c = init_cond(cond);
  return REAL(pthread_cond_timedwait)(c, mutex, ts);
}

INTERCEPTOR(int, pthread_cond_destroy, pthread_cond_t *cond) {
  __rtsan_notify_intercepted_call("pthread_cond_destroy");
  pthread_cond_t *c = init_cond(cond);
  int res = REAL(pthread_cond_destroy)(c);
  destroy_cond(c);
  return res;
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

#if SANITIZER_LINUX
INTERCEPTOR(int, sched_getaffinity, pid_t pid, size_t len, cpu_set_t *set) {
  __rtsan_notify_intercepted_call("sched_getaffinity");
  return REAL(sched_getaffinity)(pid, len, set);
}

INTERCEPTOR(int, sched_setaffinity, pid_t pid, size_t len,
            const cpu_set_t *set) {
  __rtsan_notify_intercepted_call("sched_setaffinity");
  return REAL(sched_setaffinity)(pid, len, set);
}
#define RTSAN_MAYBE_INTERCEPT_SCHED_GETAFFINITY                                \
  INTERCEPT_FUNCTION(sched_getaffinity)
#define RTSAN_MAYBE_INTERCEPT_SCHED_SETAFFINITY                                \
  INTERCEPT_FUNCTION(sched_setaffinity)
#else
#define RTSAN_MAYBE_INTERCEPT_SCHED_GETAFFINITY
#define RTSAN_MAYBE_INTERCEPT_SCHED_SETAFFINITY
#endif

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

#if SANITIZER_INTERCEPT_FREE_SIZED
INTERCEPTOR(void, free_sized, void *ptr, SIZE_T size) {
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);

  // According to the C and C++ standard, freeing a nullptr is guaranteed to be
  // a no-op (and thus real-time safe). This can be confirmed for looking at
  // __libc_free in the glibc source.
  if (ptr != nullptr)
    __rtsan_notify_intercepted_call("free_sized");

  if (REAL(free_sized))
    return REAL(free_sized)(ptr, size);
  return REAL(free)(ptr);
}
#define RTSAN_MAYBE_INTERCEPT_FREE_SIZED INTERCEPT_FUNCTION(free_sized)
#else
#define RTSAN_MAYBE_INTERCEPT_FREE_SIZED
#endif

#if SANITIZER_INTERCEPT_FREE_ALIGNED_SIZED
INTERCEPTOR(void, free_aligned_sized, void *ptr, SIZE_T alignment,
            SIZE_T size) {
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);

  // According to the C and C++ standard, freeing a nullptr is guaranteed to be
  // a no-op (and thus real-time safe). This can be confirmed for looking at
  // __libc_free in the glibc source.
  if (ptr != nullptr)
    __rtsan_notify_intercepted_call("free_aligned_sized");

  if (REAL(free_aligned_sized))
    return REAL(free_aligned_sized)(ptr, alignment, size);
  return REAL(free)(ptr);
}
#define RTSAN_MAYBE_INTERCEPT_FREE_ALIGNED_SIZED                               \
  INTERCEPT_FUNCTION(free_aligned_sized)
#else
#define RTSAN_MAYBE_INTERCEPT_FREE_ALIGNED_SIZED
#endif

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

#if SANITIZER_LINUX
// Note that even if rtsan is ported to netbsd, it has a slighty different
// and non-variadic signature
INTERCEPTOR(void *, mremap, void *oaddr, size_t olength, size_t nlength,
            int flags, ...) {
  __rtsan_notify_intercepted_call("mremap");

  // the last optional argument is only used in this case
  // as the new page region will be assigned to. Is ignored otherwise.
  if (flags & MREMAP_FIXED) {
    va_list args;

    va_start(args, flags);
    void *naddr = va_arg(args, void *);
    va_end(args);

    return REAL(mremap)(oaddr, olength, nlength, flags, naddr);
  }

  return REAL(mremap)(oaddr, olength, nlength, flags);
}
#define RTSAN_MAYBE_INTERCEPT_MREMAP INTERCEPT_FUNCTION(mremap)
#else
#define RTSAN_MAYBE_INTERCEPT_MREMAP
#endif

INTERCEPTOR(int, munmap, void *addr, size_t length) {
  __rtsan_notify_intercepted_call("munmap");
  return REAL(munmap)(addr, length);
}

#if !SANITIZER_APPLE
INTERCEPTOR(int, madvise, void *addr, size_t length, int flag) {
  __rtsan_notify_intercepted_call("madvise");
  return REAL(madvise)(addr, length, flag);
}

INTERCEPTOR(int, posix_madvise, void *addr, size_t length, int flag) {
  __rtsan_notify_intercepted_call("posix_madvise");
  return REAL(posix_madvise)(addr, length, flag);
}
#define RTSAN_MAYBE_INTERCEPT_MADVISE INTERCEPT_FUNCTION(madvise)
#define RTSAN_MAYBE_INTERCEPT_POSIX_MADVISE INTERCEPT_FUNCTION(posix_madvise)
#else
#define RTSAN_MAYBE_INTERCEPT_MADVISE
#define RTSAN_MAYBE_INTERCEPT_POSIX_MADVISE
#endif

INTERCEPTOR(int, mprotect, void *addr, size_t length, int prot) {
  __rtsan_notify_intercepted_call("mprotect");
  return REAL(mprotect)(addr, length, prot);
}

INTERCEPTOR(int, msync, void *addr, size_t length, int flag) {
  __rtsan_notify_intercepted_call("msync");
  return REAL(msync)(addr, length, flag);
}

#if SANITIZER_APPLE
INTERCEPTOR(int, mincore, const void *addr, size_t length, char *vec) {
#else
INTERCEPTOR(int, mincore, void *addr, size_t length, unsigned char *vec) {
#endif
  __rtsan_notify_intercepted_call("mincore");
  return REAL(mincore)(addr, length, vec);
}

INTERCEPTOR(int, shm_open, const char *name, int oflag, mode_t mode) {
  __rtsan_notify_intercepted_call("shm_open");
  return REAL(shm_open)(name, oflag, mode);
}

INTERCEPTOR(int, shm_unlink, const char *name) {
  __rtsan_notify_intercepted_call("shm_unlink");
  return REAL(shm_unlink)(name);
}

#if !SANITIZER_APPLE
// is supported by freebsd too
INTERCEPTOR(int, memfd_create, const char *path, unsigned int flags) {
  __rtsan_notify_intercepted_call("memfd_create");
  return REAL(memfd_create)(path, flags);
}
#define RTSAN_MAYBE_INTERCEPT_MEMFD_CREATE INTERCEPT_FUNCTION(memfd_create)
#else
#define RTSAN_MAYBE_INTERCEPT_MEMFD_CREATE
#endif

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

#if SANITIZER_INTERCEPT_GETSOCKNAME
INTERCEPTOR(int, getsockname, int socket, struct sockaddr *sa,
            socklen_t *salen) {
  __rtsan_notify_intercepted_call("getsockname");
  return REAL(getsockname)(socket, sa, salen);
}
#define RTSAN_MAYBE_INTERCEPT_GETSOCKNAME INTERCEPT_FUNCTION(getsockname)
#else
#define RTSAN_MAYBE_INTERCEPT_GETSOCKNAME
#endif

#if SANITIZER_INTERCEPT_GETPEERNAME
INTERCEPTOR(int, getpeername, int socket, struct sockaddr *sa,
            socklen_t *salen) {
  __rtsan_notify_intercepted_call("getpeername");
  return REAL(getpeername)(socket, sa, salen);
}
#define RTSAN_MAYBE_INTERCEPT_GETPEERNAME INTERCEPT_FUNCTION(getpeername)
#else
#define RTSAN_MAYBE_INTERCEPT_GETPEERNAME
#endif

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

#if SANITIZER_INTERCEPT_SENDMMSG
#if SANITIZER_MUSL
INTERCEPTOR(int, sendmmsg, int socket, struct mmsghdr *message,
            unsigned int len, unsigned int flags) {
#else
INTERCEPTOR(int, sendmmsg, int socket, struct mmsghdr *message,
            unsigned int len, int flags) {
#endif
  __rtsan_notify_intercepted_call("sendmmsg");
  return REAL(sendmmsg)(socket, message, len, flags);
}
#define RTSAN_MAYBE_INTERCEPT_SENDMMSG INTERCEPT_FUNCTION(sendmmsg)
#else
#define RTSAN_MAYBE_INTERCEPT_SENDMMSG
#endif

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

#if SANITIZER_INTERCEPT_RECVMMSG
#if SANITIZER_MUSL
INTERCEPTOR(int, recvmmsg, int socket, struct mmsghdr *message,
            unsigned int len, unsigned int flags, struct timespec *timeout) {
#elif defined(__GLIBC_MINOR__) && __GLIBC_MINOR__ < 21
INTERCEPTOR(int, recvmmsg, int socket, struct mmsghdr *message,
            unsigned int len, int flags, const struct timespec *timeout) {
#else
INTERCEPTOR(int, recvmmsg, int socket, struct mmsghdr *message,
            unsigned int len, int flags, struct timespec *timeout) {
#endif // defined(__GLIBC_MINOR) && __GLIBC_MINOR__ < 21
  __rtsan_notify_intercepted_call("recvmmsg");
  return REAL(recvmmsg)(socket, message, len, flags, timeout);
}
#define RTSAN_MAYBE_INTERCEPT_RECVMMSG INTERCEPT_FUNCTION(recvmmsg)
#else
#define RTSAN_MAYBE_INTERCEPT_RECVMMSG
#endif

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

#if SANITIZER_INTERCEPT_GETSOCKOPT
INTERCEPTOR(int, getsockopt, int socket, int level, int option, void *value,
            socklen_t *len) {
  __rtsan_notify_intercepted_call("getsockopt");
  return REAL(getsockopt)(socket, level, option, value, len);
}

INTERCEPTOR(int, setsockopt, int socket, int level, int option,
            const void *value, socklen_t len) {
  __rtsan_notify_intercepted_call("setsockopt");
  return REAL(setsockopt)(socket, level, option, value, len);
}
#define RTSAN_MAYBE_INTERCEPT_GETSOCKOPT INTERCEPT_FUNCTION(getsockopt)
#define RTSAN_MAYBE_INTERCEPT_SETSOCKOPT INTERCEPT_FUNCTION(setsockopt)
#else
#define RTSAN_MAYBE_INTERCEPT_GETSOCKOPT
#define RTSAN_MAYBE_INTERCEPT_SETSOCKOPT
#endif

INTERCEPTOR(int, socketpair, int domain, int type, int protocol, int pair[2]) {
  __rtsan_notify_intercepted_call("socketpair");
  return REAL(socketpair)(domain, type, protocol, pair);
}

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

#if SANITIZER_LINUX
INTERCEPTOR(int, inotify_init) {
  __rtsan_notify_intercepted_call("inotify_init");
  return REAL(inotify_init)();
}

INTERCEPTOR(int, inotify_init1, int flags) {
  __rtsan_notify_intercepted_call("inotify_init1");
  return REAL(inotify_init1)(flags);
}

INTERCEPTOR(int, inotify_add_watch, int fd, const char *path, uint32_t mask) {
  __rtsan_notify_intercepted_call("inotify_add_watch");
  return REAL(inotify_add_watch)(fd, path, mask);
}

INTERCEPTOR(int, inotify_rm_watch, int fd, int wd) {
  __rtsan_notify_intercepted_call("inotify_rm_watch");
  return REAL(inotify_rm_watch)(fd, wd);
}

INTERCEPTOR(int, timerfd_create, int clockid, int flags) {
  __rtsan_notify_intercepted_call("timerfd_create");
  return REAL(timerfd_create)(clockid, flags);
}

INTERCEPTOR(int, timerfd_settime, int fd, int flags, const itimerspec *newval,
            struct itimerspec *oldval) {
  __rtsan_notify_intercepted_call("timerfd_settime");
  return REAL(timerfd_settime)(fd, flags, newval, oldval);
}

INTERCEPTOR(int, timerfd_gettime, int fd, struct itimerspec *val) {
  __rtsan_notify_intercepted_call("timerfd_gettime");
  return REAL(timerfd_gettime)(fd, val);
}

/* eventfd wrappers calls SYS_eventfd2 down the line */
INTERCEPTOR(int, eventfd, unsigned int count, int flags) {
  __rtsan_notify_intercepted_call("eventfd");
  return REAL(eventfd)(count, flags);
}
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_INIT INTERCEPT_FUNCTION(inotify_init)
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_INIT1 INTERCEPT_FUNCTION(inotify_init1)
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_ADD_WATCH                                \
  INTERCEPT_FUNCTION(inotify_add_watch)
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_RM_WATCH                                 \
  INTERCEPT_FUNCTION(inotify_rm_watch)
#define RTSAN_MAYBE_INTERCEPT_TIMERFD_CREATE INTERCEPT_FUNCTION(timerfd_create)
#define RTSAN_MAYBE_INTERCEPT_TIMERFD_SETTIME                                  \
  INTERCEPT_FUNCTION(timerfd_settime)
#define RTSAN_MAYBE_INTERCEPT_TIMERFD_GETTIME                                  \
  INTERCEPT_FUNCTION(timerfd_gettime)
#define RTSAN_MAYBE_INTERCEPT_EVENTFD INTERCEPT_FUNCTION(eventfd)
#else
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_INIT
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_INIT1
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_ADD_WATCH
#define RTSAN_MAYBE_INTERCEPT_INOTIFY_RM_WATCH
#define RTSAN_MAYBE_INTERCEPT_TIMERFD_CREATE
#define RTSAN_MAYBE_INTERCEPT_TIMERFD_SETTIME
#define RTSAN_MAYBE_INTERCEPT_TIMERFD_GETTIME
#define RTSAN_MAYBE_INTERCEPT_EVENTFD
#endif

INTERCEPTOR(int, pipe, int pipefd[2]) {
  __rtsan_notify_intercepted_call("pipe");
  return REAL(pipe)(pipefd);
}

#if !SANITIZER_APPLE
INTERCEPTOR(int, pipe2, int pipefd[2], int flags) {
  __rtsan_notify_intercepted_call("pipe2");
  return REAL(pipe2)(pipefd, flags);
}
#define RTSAN_MAYBE_INTERCEPT_PIPE2 INTERCEPT_FUNCTION(pipe2)
#else
#define RTSAN_MAYBE_INTERCEPT_PIPE2
#endif

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

#if SANITIZER_INTERCEPT_PROCESS_VM_READV
INTERCEPTOR(ssize_t, process_vm_readv, pid_t pid, const struct iovec *local_iov,
            unsigned long liovcnt, const struct iovec *remote_iov,
            unsigned long riovcnt, unsigned long flags) {
  __rtsan_notify_intercepted_call("process_vm_readv");
  return REAL(process_vm_readv)(pid, local_iov, liovcnt, remote_iov, riovcnt,
                                flags);
}

INTERCEPTOR(ssize_t, process_vm_writev, pid_t pid,
            const struct iovec *local_iov, unsigned long liovcnt,
            const struct iovec *remote_iov, unsigned long riovcnt,
            unsigned long flags) {
  __rtsan_notify_intercepted_call("process_vm_writev");
  return REAL(process_vm_writev)(pid, local_iov, liovcnt, remote_iov, riovcnt,
                                 flags);
}
#define RTSAN_MAYBE_INTERCEPT_PROCESS_VM_READV                                 \
  INTERCEPT_FUNCTION(process_vm_readv)
#define RTSAN_MAYBE_INTERCEPT_PROCESS_VM_WRITEV                                \
  INTERCEPT_FUNCTION(process_vm_writev)
#else
#define RTSAN_MAYBE_INTERCEPT_PROCESS_VM_READV
#define RTSAN_MAYBE_INTERCEPT_PROCESS_VM_WRITEV
#endif

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
  RTSAN_MAYBE_INTERCEPT_FREE_SIZED;
  RTSAN_MAYBE_INTERCEPT_FREE_ALIGNED_SIZED;
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(reallocf);
  INTERCEPT_FUNCTION(valloc);
  RTSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC;
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(mmap);
  RTSAN_MAYBE_INTERCEPT_MMAP64;
  RTSAN_MAYBE_INTERCEPT_MREMAP;
  INTERCEPT_FUNCTION(munmap);
  RTSAN_MAYBE_INTERCEPT_MADVISE;
  RTSAN_MAYBE_INTERCEPT_POSIX_MADVISE;
  INTERCEPT_FUNCTION(mprotect);
  INTERCEPT_FUNCTION(msync);
  INTERCEPT_FUNCTION(mincore);
  INTERCEPT_FUNCTION(shm_open);
  INTERCEPT_FUNCTION(shm_unlink);
  RTSAN_MAYBE_INTERCEPT_MEMFD_CREATE;
  RTSAN_MAYBE_INTERCEPT_MEMALIGN;
  RTSAN_MAYBE_INTERCEPT_PVALLOC;

  INTERCEPT_FUNCTION(open);
  RTSAN_MAYBE_INTERCEPT_OPEN64;
  INTERCEPT_FUNCTION(openat);
  RTSAN_MAYBE_INTERCEPT_OPENAT64;
  INTERCEPT_FUNCTION(close);
  INTERCEPT_FUNCTION(chdir);
  INTERCEPT_FUNCTION(fchdir);
  RTSAN_MAYBE_INTERCEPT_READLINK;
  RTSAN_MAYBE_INTERCEPT_READLINKAT;
  INTERCEPT_FUNCTION(unlink);
  INTERCEPT_FUNCTION(unlinkat);
  INTERCEPT_FUNCTION(symlink);
  INTERCEPT_FUNCTION(symlinkat);
  INTERCEPT_FUNCTION(truncate);
  INTERCEPT_FUNCTION(ftruncate);
  RTSAN_MAYBE_INTERCEPT_TRUNCATE64;
  RTSAN_MAYBE_INTERCEPT_FTRUNCATE64;
  INTERCEPT_FUNCTION(fopen);
  RTSAN_MAYBE_INTERCEPT_FOPEN64;
  RTSAN_MAYBE_INTERCEPT_FREOPEN64;
  INTERCEPT_FUNCTION(fread);
  INTERCEPT_FUNCTION(read);
  INTERCEPT_FUNCTION(write);
  INTERCEPT_FUNCTION(pread);
  RTSAN_MAYBE_INTERCEPT_PREAD64;
  RTSAN_MAYBE_INTERCEPT_PREADV;
  RTSAN_MAYBE_INTERCEPT_PREADV64;
  INTERCEPT_FUNCTION(readv);
  INTERCEPT_FUNCTION(pwrite);
  RTSAN_MAYBE_INTERCEPT_PWRITE64;
  RTSAN_MAYBE_INTERCEPT_PWRITEV;
  RTSAN_MAYBE_INTERCEPT_PWRITEV64;
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
  RTSAN_MAYBE_INTERCEPT_FPURGE;
  RTSAN_MAYBE_INTERCEPT_PIPE2;
  INTERCEPT_FUNCTION(fdopen);
  INTERCEPT_FUNCTION(freopen);
  RTSAN_MAYBE_INTERCEPT_FOPENCOOKIE;
  RTSAN_MAYBE_INTERCEPT_OPEN_MEMSTREAM;
  RTSAN_MAYBE_INTERCEPT_FMEMOPEN;
  RTSAN_MAYBE_INTERCEPT_SETBUF;
  RTSAN_MAYBE_INTERCEPT_SETVBUF;
  RTSAN_MAYBE_INTERCEPT_SETLINEBUF;
  RTSAN_MAYBE_INTERCEPT_SETBUFFER;
  RTSAN_MAYBE_INTERCEPT_FGETPOS;
  RTSAN_MAYBE_INTERCEPT_FSEEK;
  RTSAN_MAYBE_INTERCEPT_FSEEKO;
  RTSAN_MAYBE_INTERCEPT_FSETPOS;
  RTSAN_MAYBE_INTERCEPT_FTELL;
  RTSAN_MAYBE_INTERCEPT_FTELLO;
  RTSAN_MAYBE_INTERCEPT_REWIND;
  RTSAN_MAYBE_INTERCEPT_FGETPOS64;
  RTSAN_MAYBE_INTERCEPT_FSEEKO64;
  RTSAN_MAYBE_INTERCEPT_FSETPOS64;
  RTSAN_MAYBE_INTERCEPT_FTELLO64;
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

  // See the comment in tsan_interceptors_posix.cpp.
#if SANITIZER_GLIBC && !__GLIBC_PREREQ(2, 36) &&                               \
    (defined(__x86_64__) || defined(__mips__) || SANITIZER_PPC64V1 ||          \
     defined(__s390x__))
  INTERCEPT_FUNCTION_VER(pthread_cond_init, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_signal, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_broadcast, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_wait, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_timedwait, "GLIBC_2.3.2");
  INTERCEPT_FUNCTION_VER(pthread_cond_destroy, "GLIBC_2.3.2");
#else
  INTERCEPT_FUNCTION(pthread_cond_init);
  INTERCEPT_FUNCTION(pthread_cond_signal);
  INTERCEPT_FUNCTION(pthread_cond_broadcast);
  INTERCEPT_FUNCTION(pthread_cond_wait);
  INTERCEPT_FUNCTION(pthread_cond_timedwait);
  INTERCEPT_FUNCTION(pthread_cond_destroy);
#endif

  INTERCEPT_FUNCTION(pthread_rwlock_rdlock);
  INTERCEPT_FUNCTION(pthread_rwlock_unlock);
  INTERCEPT_FUNCTION(pthread_rwlock_wrlock);

  INTERCEPT_FUNCTION(sleep);
  INTERCEPT_FUNCTION(usleep);
  INTERCEPT_FUNCTION(nanosleep);
  INTERCEPT_FUNCTION(sched_yield);
  RTSAN_MAYBE_INTERCEPT_SCHED_GETAFFINITY;
  RTSAN_MAYBE_INTERCEPT_SCHED_SETAFFINITY;

  INTERCEPT_FUNCTION(accept);
  INTERCEPT_FUNCTION(bind);
  INTERCEPT_FUNCTION(connect);
  INTERCEPT_FUNCTION(getaddrinfo);
  INTERCEPT_FUNCTION(getnameinfo);
  INTERCEPT_FUNCTION(listen);
  INTERCEPT_FUNCTION(recv);
  INTERCEPT_FUNCTION(recvfrom);
  INTERCEPT_FUNCTION(recvmsg);
  RTSAN_MAYBE_INTERCEPT_RECVMMSG;
  INTERCEPT_FUNCTION(send);
  INTERCEPT_FUNCTION(sendmsg);
  RTSAN_MAYBE_INTERCEPT_SENDMMSG;
  INTERCEPT_FUNCTION(sendto);
  INTERCEPT_FUNCTION(shutdown);
  INTERCEPT_FUNCTION(socket);
  RTSAN_MAYBE_INTERCEPT_ACCEPT4;
  RTSAN_MAYBE_INTERCEPT_GETSOCKNAME;
  RTSAN_MAYBE_INTERCEPT_GETPEERNAME;
  RTSAN_MAYBE_INTERCEPT_GETSOCKOPT;
  RTSAN_MAYBE_INTERCEPT_SETSOCKOPT;
  INTERCEPT_FUNCTION(socketpair);

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

  RTSAN_MAYBE_INTERCEPT_INOTIFY_INIT;
  RTSAN_MAYBE_INTERCEPT_INOTIFY_INIT1;
  RTSAN_MAYBE_INTERCEPT_INOTIFY_ADD_WATCH;
  RTSAN_MAYBE_INTERCEPT_INOTIFY_RM_WATCH;

  RTSAN_MAYBE_INTERCEPT_TIMERFD_CREATE;
  RTSAN_MAYBE_INTERCEPT_TIMERFD_SETTIME;
  RTSAN_MAYBE_INTERCEPT_TIMERFD_GETTIME;
  RTSAN_MAYBE_INTERCEPT_EVENTFD;

  INTERCEPT_FUNCTION(pipe);
  INTERCEPT_FUNCTION(mkfifo);

  INTERCEPT_FUNCTION(fork);
  INTERCEPT_FUNCTION(execve);

  RTSAN_MAYBE_INTERCEPT_PROCESS_VM_READV;
  RTSAN_MAYBE_INTERCEPT_PROCESS_VM_WRITEV;

  INTERCEPT_FUNCTION(syscall);
}

#endif // SANITIZER_POSIX
