//===-- esan_interceptors.cpp ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Interception routines for the esan run-time.
//===----------------------------------------------------------------------===//

#include "esan.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __esan; // NOLINT

#define CUR_PC() (StackTrace::GetCurrentPc())

//===----------------------------------------------------------------------===//
// Interception via sanitizer common interceptors
//===----------------------------------------------------------------------===//

// Get the per-platform defines for what is possible to intercept
#include "sanitizer_common/sanitizer_platform_interceptors.h"

// TODO(bruening): tsan disables several interceptors (getpwent, etc.) claiming
// that interception is a perf hit: should we do the same?

// We have no need to intercept:
#undef SANITIZER_INTERCEPT_TLS_GET_ADDR

// TODO(bruening): the common realpath interceptor assumes malloc is
// intercepted!  We should try to parametrize that, though we'll
// intercept malloc soon ourselves and can then remove this undef.
#undef SANITIZER_INTERCEPT_REALPATH

#define COMMON_INTERCEPTOR_NOTHING_IS_INITIALIZED (!EsanIsInitialized)

#define COMMON_INTERCEPT_FUNCTION(name) INTERCEPT_FUNCTION(name)

#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...)                               \
  do {                                                                         \
    if (UNLIKELY(COMMON_INTERCEPTOR_NOTHING_IS_INITIALIZED)) {                 \
      return REAL(func)(__VA_ARGS__);                                          \
    }                                                                          \
    (void)ctx;                                                                 \
  } while (false)

#define COMMON_INTERCEPTOR_ENTER_NOIGNORE(ctx, func, ...)                      \
  COMMON_INTERCEPTOR_ENTER(ctx, func, __VA_ARGS__)

#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size)                         \
  processRangeAccess(CUR_PC(), (uptr)ptr, size, true)

#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size)                          \
  processRangeAccess(CUR_PC(), (uptr)ptr, size, false)

// This is only called if the app explicitly calls exit(), not on
// a normal exit.
#define COMMON_INTERCEPTOR_ON_EXIT(ctx) finalizeLibrary()

#define COMMON_INTERCEPTOR_FILE_OPEN(ctx, file, path) {}
#define COMMON_INTERCEPTOR_FILE_CLOSE(ctx, file) {}
#define COMMON_INTERCEPTOR_LIBRARY_LOADED(filename, handle) {}
#define COMMON_INTERCEPTOR_LIBRARY_UNLOADED() {}
#define COMMON_INTERCEPTOR_ACQUIRE(ctx, u) {}
#define COMMON_INTERCEPTOR_RELEASE(ctx, u) {}
#define COMMON_INTERCEPTOR_DIR_ACQUIRE(ctx, path) {}
#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) {}
#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) {}
#define COMMON_INTERCEPTOR_FD_ACCESS(ctx, fd) {}
#define COMMON_INTERCEPTOR_FD_SOCKET_ACCEPT(ctx, fd, newfd) {}
#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) {}
#define COMMON_INTERCEPTOR_SET_PTHREAD_NAME(ctx, thread, name) {}
#define COMMON_INTERCEPTOR_BLOCK_REAL(name) REAL(name)
#define COMMON_INTERCEPTOR_MUTEX_LOCK(ctx, m) {}
#define COMMON_INTERCEPTOR_MUTEX_UNLOCK(ctx, m) {}
#define COMMON_INTERCEPTOR_MUTEX_REPAIR(ctx, m) {}
#define COMMON_INTERCEPTOR_HANDLE_RECVMSG(ctx, msg) {}
#define COMMON_INTERCEPTOR_USER_CALLBACK_START() {}
#define COMMON_INTERCEPTOR_USER_CALLBACK_END() {}

#include "sanitizer_common/sanitizer_common_interceptors.inc"

//===----------------------------------------------------------------------===//
// Syscall interception
//===----------------------------------------------------------------------===//

// We want the caller's PC b/c unlike the other function interceptors these
// are separate pre and post functions called around the app's syscall().

#define COMMON_SYSCALL_PRE_READ_RANGE(ptr, size)                               \
  processRangeAccess(GET_CALLER_PC(), (uptr)ptr, size, false)

#define COMMON_SYSCALL_PRE_WRITE_RANGE(ptr, size) {}

#define COMMON_SYSCALL_POST_READ_RANGE(ptr, size) {}

// The actual amount written is in post, not pre.
#define COMMON_SYSCALL_POST_WRITE_RANGE(ptr, size)                             \
  processRangeAccess(GET_CALLER_PC(), (uptr)ptr, size, true)

#define COMMON_SYSCALL_ACQUIRE(addr) {}
#define COMMON_SYSCALL_RELEASE(addr) { (void)addr; }
#define COMMON_SYSCALL_FD_CLOSE(fd) {}
#define COMMON_SYSCALL_FD_ACQUIRE(fd) {}
#define COMMON_SYSCALL_FD_RELEASE(fd) {}
#define COMMON_SYSCALL_PRE_FORK() {}
#define COMMON_SYSCALL_POST_FORK(res) {}

#include "sanitizer_common/sanitizer_common_syscalls.inc"

//===----------------------------------------------------------------------===//
// Custom interceptors
//===----------------------------------------------------------------------===//

// TODO(bruening): move more of these to the common interception pool as they
// are shared with tsan and asan.
// While our other files match LLVM style, here we match sanitizer style as we
// expect to move these to the common pool.

INTERCEPTOR(char *, strcpy, char *dst, const char *src) { // NOLINT
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strcpy, dst, src);
  uptr srclen = internal_strlen(src);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dst, srclen + 1);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, src, srclen + 1);
  return REAL(strcpy)(dst, src); // NOLINT
}

INTERCEPTOR(char *, strncpy, char *dst, char *src, uptr n) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, strncpy, dst, src, n);
  uptr srclen = internal_strnlen(src, n);
  uptr copied_size = srclen + 1 > n ? n : srclen + 1;
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, dst, copied_size);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, src, copied_size);
  return REAL(strncpy)(dst, src, n);
}

#if SANITIZER_FREEBSD || SANITIZER_MAC || SANITIZER_ANDROID
INTERCEPTOR(int, stat, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, stat, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(stat)(path, buf);
#define ESAN_INTERCEPT_STAT INTERCEPT_FUNCTION(stat)
#else
INTERCEPTOR(int, __xstat, int version, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __xstat, version, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(__xstat)(version, path, buf);
}
#define ESAN_INTERCEPT_STAT INTERCEPT_FUNCTION(__xstat)
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID && !SANITIZER_FREEBSD
INTERCEPTOR(int, __xstat64, int version, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __xstat64, version, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(__xstat64)(version, path, buf);
}
#define ESAN_MAYBE_INTERCEPT___XSTAT64 INTERCEPT_FUNCTION(__xstat64)
#else
#define ESAN_MAYBE_INTERCEPT___XSTAT64
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID && !SANITIZER_FREEBSD
INTERCEPTOR(int, stat64, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, stat64, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(stat64)(path, buf);
}
#define ESAN_MAYBE_INTERCEPT_STAT64 INTERCEPT_FUNCTION(stat64)
#else
#define ESAN_MAYBE_INTERCEPT_STAT64
#endif

#if SANITIZER_FREEBSD || SANITIZER_MAC || SANITIZER_ANDROID
INTERCEPTOR(int, lstat, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lstat, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(lstat)(path, buf);
}
#define ESAN_INTERCEPT_LSTAT INTERCEPT_FUNCTION(lstat)
#else
INTERCEPTOR(int, __lxstat, int version, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __lxstat, version, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(__lxstat)(version, path, buf);
}
#define ESAN_INTERCEPT_LSTAT INTERCEPT_FUNCTION(__lxstat)
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID && !SANITIZER_FREEBSD
INTERCEPTOR(int, __lxstat64, int version, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, __lxstat64, version, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(__lxstat64)(version, path, buf);
}
#define ESAN_MAYBE_INTERCEPT___LXSTAT64 INTERCEPT_FUNCTION(__lxstat64)
#else
#define ESAN_MAYBE_INTERCEPT___LXSTAT64
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID && !SANITIZER_FREEBSD
INTERCEPTOR(int, lstat64, const char *path, void *buf) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, lstat64, path, buf);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(lstat64)(path, buf);
}
#define ESAN_MAYBE_INTERCEPT_LSTAT64 INTERCEPT_FUNCTION(lstat64)
#else
#define ESAN_MAYBE_INTERCEPT_LSTAT64
#endif

INTERCEPTOR(int, open, const char *name, int flags, int mode) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, open, name, flags, mode);
  COMMON_INTERCEPTOR_READ_STRING(ctx, name, 0);
  return REAL(open)(name, flags, mode);
}

#if SANITIZER_LINUX
INTERCEPTOR(int, open64, const char *name, int flags, int mode) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, open64, name, flags, mode);
  COMMON_INTERCEPTOR_READ_STRING(ctx, name, 0);
  return REAL(open64)(name, flags, mode);
}
#define ESAN_MAYBE_INTERCEPT_OPEN64 INTERCEPT_FUNCTION(open64)
#else
#define ESAN_MAYBE_INTERCEPT_OPEN64
#endif

INTERCEPTOR(int, creat, const char *name, int mode) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, creat, name, mode);
  COMMON_INTERCEPTOR_READ_STRING(ctx, name, 0);
  return REAL(creat)(name, mode);
}

#if SANITIZER_LINUX
INTERCEPTOR(int, creat64, const char *name, int mode) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, creat64, name, mode);
  COMMON_INTERCEPTOR_READ_STRING(ctx, name, 0);
  return REAL(creat64)(name, mode);
}
#define ESAN_MAYBE_INTERCEPT_CREAT64 INTERCEPT_FUNCTION(creat64)
#else
#define ESAN_MAYBE_INTERCEPT_CREAT64
#endif

INTERCEPTOR(int, unlink, char *path) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, unlink, path);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(unlink)(path);
}

INTERCEPTOR(uptr, fread, void *ptr, uptr size, uptr nmemb, void *f) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fread, ptr, size, nmemb, f);
  COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size * nmemb);
  return REAL(fread)(ptr, size, nmemb, f);
}

INTERCEPTOR(uptr, fwrite, const void *p, uptr size, uptr nmemb, void *f) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, fwrite, p, size, nmemb, f);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, p, size * nmemb);
  return REAL(fwrite)(p, size, nmemb, f);
}

INTERCEPTOR(int, puts, const char *s) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, puts, s);
  COMMON_INTERCEPTOR_READ_RANGE(ctx, s, internal_strlen(s));
  return REAL(puts)(s);
}

INTERCEPTOR(int, rmdir, char *path) {
  void *ctx;
  COMMON_INTERCEPTOR_ENTER(ctx, rmdir, path);
  COMMON_INTERCEPTOR_READ_STRING(ctx, path, 0);
  return REAL(rmdir)(path);
}

namespace __esan {

void initializeInterceptors() {
  InitializeCommonInterceptors();

  INTERCEPT_FUNCTION(strcpy); // NOLINT
  INTERCEPT_FUNCTION(strncpy);

  ESAN_INTERCEPT_STAT;
  ESAN_MAYBE_INTERCEPT_STAT64;
  ESAN_MAYBE_INTERCEPT___XSTAT64;
  ESAN_INTERCEPT_LSTAT;
  ESAN_MAYBE_INTERCEPT_LSTAT64;
  ESAN_MAYBE_INTERCEPT___LXSTAT64;
  INTERCEPT_FUNCTION(open);
  ESAN_MAYBE_INTERCEPT_OPEN64;
  INTERCEPT_FUNCTION(creat);
  ESAN_MAYBE_INTERCEPT_CREAT64;
  INTERCEPT_FUNCTION(unlink);
  INTERCEPT_FUNCTION(fread);
  INTERCEPT_FUNCTION(fwrite);
  INTERCEPT_FUNCTION(puts);
  INTERCEPT_FUNCTION(rmdir);

  // TODO(bruening): we should intercept calloc() and other memory allocation
  // routines that zero memory and update our shadow memory appropriately.

  // TODO(bruening): intercept routines that other sanitizers intercept that
  // are not in the common pool or here yet, ideally by adding to the common
  // pool.  Examples include wcslen and bcopy.

  // TODO(bruening): there are many more libc routines that read or write data
  // structures that no sanitizer is intercepting: sigaction, strtol, etc.
}

} // namespace __esan
