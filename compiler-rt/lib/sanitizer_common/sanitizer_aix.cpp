//===-- sanitizer_aix.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries and
// implements AIX-specific functions.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_AIX
#  include <dlfcn.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <pthread.h>
#  include <sched.h>
#  include <signal.h>
#  include <stdlib.h>
#  include <string.h>
#  include <sys/errno.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <sys/time.h>
#  include <sys/types.h>
#  include <unistd.h>

#  include "sanitizer_aix.h"
#  include "sanitizer_common.h"
#  include "sanitizer_libc.h"

namespace __sanitizer {

#  include "sanitizer_syscall_generic.inc"

static void* GetFuncAddr(const char* name) {
  // FIXME: if we are going to ship dynamic asan library, we may need to search
  // all the loaded modules with RTLD_DEFAULT if RTLD_NEXT failed.
  void* addr = dlsym(RTLD_NEXT, name);
  return addr;
}

// Internal implementation for the libc functions are also calling to the same
// name function in libc. However because the same name function to libc may be
// intercepted, and in the interceptor function, it may call REAL(func). But the
// REAL(func) may be not assigned at this time, because internal_func may be
// called before interceptor init functions are called. So we need to call to
// libc function via function pointer.

//
// The function pointer has the type declared by the libc headers. The headers
// may map a function to another symbol (e.g., mmap to mmap64 when _LARGE_FILES
// is defined), so the symbol is looked up by the name that `func` expands to.

#  define _REAL(func, ...) real_##func(__VA_ARGS__)

#  define DEFINE__REAL(func)                             \
    static decltype(&::func) real_##func = nullptr;      \
    if (!real_##func) {                                  \
      real_##func = reinterpret_cast<decltype(&::func)>( \
          GetFuncAddr(SANITIZER_STRINGIFY(func)));       \
    }                                                    \
    CHECK(real_##func)

uptr internal_mmap(void* addr, uptr length, int prot, int flags, int fd,
                   u64 offset) {
  DEFINE__REAL(mmap);
  return reinterpret_cast<uptr>(
      _REAL(mmap, addr, length, prot, flags, fd, offset));
}

uptr internal_munmap(void* addr, uptr length) {
  DEFINE__REAL(munmap);
  return _REAL(munmap, addr, length);
}

int internal_mprotect(void* addr, uptr length, int prot) {
  DEFINE__REAL(mprotect);
  return _REAL(mprotect, addr, length, prot);
}

int internal_madvise(uptr addr, uptr length, int advice) {
  DEFINE__REAL(madvise);
  return _REAL(madvise, reinterpret_cast<caddr_t>(addr), length, advice);
}

uptr internal_close(fd_t fd) {
  DEFINE__REAL(close);
  return _REAL(close, fd);
}

uptr internal_open(const char* filename, int flags) {
  DEFINE__REAL(open);
  return _REAL(open, filename, flags);
}

uptr internal_open(const char* filename, int flags, u32 mode) {
  DEFINE__REAL(open);
  return _REAL(open, filename, flags, mode);
}

uptr internal_read(fd_t fd, void* buf, uptr count) {
  DEFINE__REAL(read);
  return _REAL(read, fd, buf, count);
}

uptr internal_write(fd_t fd, const void* buf, uptr count) {
  DEFINE__REAL(write);
  return _REAL(write, fd, buf, count);
}

uptr internal_stat(const char* path, void* buf) {
  DEFINE__REAL(stat);
  return _REAL(stat, path, static_cast<struct stat*>(buf));
}

uptr internal_lstat(const char* path, void* buf) {
  DEFINE__REAL(lstat);
  return _REAL(lstat, path, static_cast<struct stat*>(buf));
}

uptr internal_fstat(fd_t fd, void* buf) {
  DEFINE__REAL(fstat);
  return _REAL(fstat, fd, static_cast<struct stat*>(buf));
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup(int oldfd) {
  DEFINE__REAL(dup);
  return _REAL(dup, oldfd);
}

uptr internal_dup2(int oldfd, int newfd) {
  DEFINE__REAL(dup2);
  return _REAL(dup2, oldfd, newfd);
}

uptr internal_readlink(const char* path, char* buf, uptr bufsize) {
  DEFINE__REAL(readlink);
  return _REAL(readlink, path, buf, bufsize);
}

uptr internal_unlink(const char* path) {
  DEFINE__REAL(unlink);
  return _REAL(unlink, path);
}

uptr internal_sched_yield() {
  DEFINE__REAL(sched_yield);
  return _REAL(sched_yield);
}

void internal__exit(int exitcode) {
  DEFINE__REAL(_exit);
  _REAL(_exit, exitcode);
  Die();  // Unreachable.
}

void internal_usleep(u64 useconds) {
  DEFINE__REAL(usleep);
  _REAL(usleep, useconds);
}

uptr internal_getpid() {
  DEFINE__REAL(getpid);
  return _REAL(getpid);
}

int internal_sigaction(int signum, const void* act, void* oldact) {
  DEFINE__REAL(sigaction);
  return _REAL(sigaction, signum, static_cast<const struct sigaction*>(act),
               static_cast<struct sigaction*>(oldact));
}

void internal_sigfillset(__sanitizer_sigset_t* set) {
  DEFINE__REAL(sigfillset);
  _REAL(sigfillset, reinterpret_cast<sigset_t*>(set));
}

uptr internal_sigprocmask(int how, __sanitizer_sigset_t* set,
                          __sanitizer_sigset_t* oldset) {
  DEFINE__REAL(sigprocmask);
  return _REAL(sigprocmask, how, reinterpret_cast<sigset_t*>(set),
               reinterpret_cast<sigset_t*>(oldset));
}

int internal_fork() {
  DEFINE__REAL(fork);
  return _REAL(fork);
}

uptr internal_execve(const char* filename, char* const argv[],
                     char* const envp[]) {
  DEFINE__REAL(execve);
  return _REAL(execve, filename, argv, envp);
}

uptr internal_waitpid(int pid, int* status, int options) {
  DEFINE__REAL(waitpid);
  return _REAL(waitpid, pid, status, options);
}

void* internal_start_thread(void* (*func)(void* arg), void* arg) {
  if (&internal_pthread_create == 0)
    return nullptr;
  // Start the thread with signals blocked, otherwise it can steal user signals.
  __sanitizer_sigset_t set, old;
  internal_sigfillset(&set);
  internal_sigprocmask(SIG_SETMASK, &set, &old);
  // pthread_t is an unsigned int on AIX, which is narrower than void* in 64-bit
  // mode, so it cannot be stored to a void* directly.
  pthread_t th;
  internal_pthread_create(&th, nullptr, func, arg);
  internal_sigprocmask(SIG_SETMASK, &old, nullptr);
  return reinterpret_cast<void*>(static_cast<uptr>(th));
}

void internal_join_thread(void* th) {
  if (&internal_pthread_join)
    internal_pthread_join(th, nullptr);
}

uptr internal_clock_gettime(__sanitizer_clockid_t clk_id, void* tp) {
  DEFINE__REAL(clock_gettime);
  return _REAL(clock_gettime, static_cast<clockid_t>(clk_id),
               static_cast<struct timespec*>(tp));
}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
