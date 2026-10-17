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
#  include <stdio.h>
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

#  define _REAL(func, ...) real##_##func(__VA_ARGS__)

#  define DEFINE__REAL(ret_type, func, ...)                        \
    static ret_type (*real_##func)(__VA_ARGS__) = NULL;            \
    if (!real_##func) {                                            \
      real_##func = (ret_type (*)(__VA_ARGS__))GetFuncAddr(#func); \
    }                                                              \
    CHECK(real_##func);

uptr internal_mmap(void* addr, uptr length, int prot, int flags, int fd,
                   u64 offset) {
  DEFINE__REAL(uptr, mmap, void* addr, uptr length, int prot, int flags, int fd,
               u64 offset);
  return _REAL(mmap, addr, length, prot, flags, fd, offset);
}

uptr internal_munmap(void* addr, uptr length) {
  DEFINE__REAL(uptr, munmap, void* addr, uptr length);
  return _REAL(munmap, addr, length);
}

int internal_mprotect(void* addr, uptr length, int prot) {
  DEFINE__REAL(int, mprotect, void* addr, uptr length, int prot);
  return _REAL(mprotect, addr, length, prot);
}

int internal_madvise(uptr addr, uptr length, int advice) {
  char* raddr = reinterpret_cast<char*>(addr);
  DEFINE__REAL(int, madvise, char* raddr, uptr length, int advice)
  return _REAL(madvise, raddr, length, advice);
}

uptr internal_close(fd_t fd) {
  DEFINE__REAL(uptr, close, fd_t fd);
  return _REAL(close, fd);
}

uptr internal_open(const char* filename, int flags) {
  DEFINE__REAL(uptr, open, const char* filename, int flags);
  return _REAL(open, filename, flags);
}

uptr internal_open(const char* filename, int flags, u32 mode) {
  DEFINE__REAL(uptr, open, const char* filename, int flags, u32 mode);
  return _REAL(open, filename, flags, mode);
}

__sanitizer_FILE* internal_popen(const char* command, const char* type) {
  DEFINE__REAL(__sanitizer_FILE*, popen, const char* command, const char* type);
  return _REAL(popen, command, type);
}

int internal_pclose(__sanitizer_FILE* file) {
  FILE* rfile = reinterpret_cast<FILE*>(file);
  DEFINE__REAL(int, pclose, FILE* file);
  return _REAL(pclose, rfile);
}

uptr internal_read(fd_t fd, void* buf, uptr count) {
  DEFINE__REAL(uptr, read, fd_t fd, void* buf, uptr count);
  return _REAL(read, fd, buf, count);
}

uptr internal_write(fd_t fd, const void* buf, uptr count) {
  DEFINE__REAL(uptr, write, fd_t fd, const void* buf, uptr count);
  return _REAL(write, fd, buf, count);
}

uptr internal_stat(const char* path, void* buf) {
  DEFINE__REAL(uptr, stat, const char* path, void* buf);
  return _REAL(stat, path, buf);
}

uptr internal_lstat(const char* path, void* buf) {
  DEFINE__REAL(uptr, lstat, const char* path, void* buf);
  return _REAL(lstat, path, buf);
}

uptr internal_fstat(fd_t fd, void* buf) {
  DEFINE__REAL(uptr, fstat, fd_t fd, void* buf);
  return _REAL(fstat, fd, buf);
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup(int oldfd) {
  DEFINE__REAL(uptr, dup, int oldfd);
  return _REAL(dup, oldfd);
}

uptr internal_dup2(int oldfd, int newfd) {
  DEFINE__REAL(uptr, dup2, int oldfd, int newfd);
  return _REAL(dup2, oldfd, newfd);
}

uptr internal_readlink(const char* path, char* buf, uptr bufsize) {
  DEFINE__REAL(uptr, readlink, const char* path, char* buf, uptr bufsize);
  return _REAL(readlink, path, buf, bufsize);
}

uptr internal_unlink(const char* path) {
  DEFINE__REAL(uptr, unlink, const char* path);
  return _REAL(unlink, path);
}

uptr internal_sched_yield() {
  DEFINE__REAL(uptr, sched_yield);
  return _REAL(sched_yield);
}

void internal__exit(int exitcode) {
  DEFINE__REAL(void, _exit, int exitcode);
  _REAL(_exit, exitcode);
  Die();  // Unreachable.
}

void internal_usleep(u64 useconds) {
  DEFINE__REAL(void, usleep, u64 useconds);
  _REAL(usleep, useconds);
}

uptr internal_getpid() {
  DEFINE__REAL(uptr, getpid);
  return _REAL(getpid);
}

int internal_sigaction(int signum, const void* act, void* oldact) {
  DEFINE__REAL(int, sigaction, int signum, const void* act, void* oldact);
  return _REAL(sigaction, signum, act, oldact);
}

void internal_sigfillset(__sanitizer_sigset_t* set) {
  sigset_t* rset = reinterpret_cast<sigset_t*>(set);
  DEFINE__REAL(void, sigfillset, sigset_t* rset);
  _REAL(sigfillset, rset);
}

uptr internal_sigprocmask(int how, __sanitizer_sigset_t* set,
                          __sanitizer_sigset_t* oldset) {
  sigset_t* rset = reinterpret_cast<sigset_t*>(set);
  sigset_t* roldset = reinterpret_cast<sigset_t*>(oldset);
  DEFINE__REAL(uptr, sigprocmask, int how, sigset_t* rset, sigset_t* roldset);
  return _REAL(sigprocmask, how, rset, roldset);
}

char* internal_getcwd(char* buf, uptr size) {
  DEFINE__REAL(char*, getcwd, char* buf, uptr size);
  return _REAL(getcwd, buf, size);
}

int internal_fork() {
  DEFINE__REAL(int, fork);
  return _REAL(fork);
}

uptr internal_execve(const char* filename, char* const argv[],
                     char* const envp[]) {
  DEFINE__REAL(uptr, execve, const char* filename, char* const argv[],
               char* const envp[]);
  return _REAL(execve, filename, argv, envp);
}

uptr internal_waitpid(int pid, int* status, int options) {
  DEFINE__REAL(uptr, waitpid, int pid, int* status, int options);
  return _REAL(waitpid, pid, status, options);
}

int internal_pthread_join(pthread_t thread, void** status) {
  DEFINE__REAL(int, pthread_join, pthread_t thread, void** status);
  return _REAL(pthread_join, thread, status);
}

int internal_pthread_create(pthread_t* thread, const pthread_attr_t* attr,
                            void* (*start_routine)(void*), void* arg) {
  DEFINE__REAL(int, pthread_create, pthread_t* thread,
               const pthread_attr_t* attr, void* (*start_routine)(void*),
               void* arg);
  return _REAL(pthread_create, thread, attr, start_routine, arg);
}

void* internal_start_thread(void* (*func)(void* arg), void* arg) {
  // Start the thread with signals blocked, otherwise it can steal user signals.
  __sanitizer_sigset_t set, old;
  internal_sigfillset(&set);
  internal_sigprocmask(SIG_SETMASK, &set, &old);
  pthread_t th;
  internal_pthread_create(&th, 0, func, arg);
  internal_sigprocmask(SIG_SETMASK, &old, 0);
  // pthread_t is unsigned int on AIX
  return reinterpret_cast<void*>(th);
}

void internal_join_thread(void* th) {
  internal_pthread_join((pthread_t)(reinterpret_cast<uptr>(th)), nullptr);
}

uptr internal_clock_gettime(__sanitizer_clockid_t clk_id, void* tp) {
  clock_t rclk_id = reinterpret_cast<clock_t>(clk_id);
  struct timespec* rtp = reinterpret_cast<struct timespec*>(tp);
  DEFINE__REAL(uptr, clock_gettime, clock_t rclk_id, struct timespec* rtp);
  return _REAL(clock_gettime, rclk_id, rtp);
}

}  // namespace __sanitizer

#endif  // SANITIZER_AIX
