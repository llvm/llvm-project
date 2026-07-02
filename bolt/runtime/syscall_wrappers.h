//===- bolt/runtime/syscall_wrappers.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_SYSCALL_WRAPPERS
#define LLVM_TOOLS_LLVM_BOLT_SYSCALL_WRAPPERS

#if defined(__aarch64__) || defined(__arm64__)
#include "sys_aarch64.h"
#elif defined(__riscv)
#include "sys_riscv64.h"
#elif defined(__x86_64__)
#include "sys_x86_64.h"
#else
#error "For AArch64/ARM64,X86_64 AND RISCV64 only."
#endif

#include "runtime_types.h"

// LLVM's libc syscall wrappers
#define LIBC_NAMESPACE __bolt_runtime_syscall_wrappres
#include "src/__support/OSUtil/linux/syscall.h"

// Anonymous namespace covering everything but our library entry point
namespace {

// Declare some syscall wrappers we use throughout this code to avoid linking
// against system libc.
// Syscall wrapper return and argument types are based on LLVM's libc.

// Reference: libc/src/unistd/linux/read.cpp
ssize_t __read(int fd, void *buf, size_t count) {
  return LIBC_NAMESPACE::syscall_impl<ssize_t>(__NR_read, fd, buf, count);
}

// Reference: libc/src/unistd/linux/write.cpp
ssize_t __write(int fd, const void *buf, size_t count) {
  return LIBC_NAMESPACE::syscall_impl<ssize_t>(__NR_write, fd, buf, count);
}

// Reference: libc/src/sys/mman/linux/mmap.cpp
void *__mmap(void *addr, size_t size, int prot, int flags, int fd,
             off_t offset) {
  return LIBC_NAMESPACE::syscall_impl<void *>(__NR_mmap, addr, size, prot,
                                              flags, fd, offset);
}

// Reference: libc/src/sys/mman/linux/munmap.cpp
int __munmap(void *addr, size_t size) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_munmap, addr, size);
}

// Reference: libc/src/signal/linux/sigprocmask.cpp
int __sigprocmask(int how, const sigset_t *set, sigset_t *oldset) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_rt_sigprocmask, how, set,
                                           oldset, sizeof(sigset_t));
}

// Reference: libc/src/unistd/linux/getpid.cpp
pid_t __getpid() { return LIBC_NAMESPACE::syscall_impl<pid_t>(__NR_getpid); }

// Reference: libc/src/__support/OSUtil/linux/exit.cpp
#if defined(__x86_64__)
__attribute__((no_stack_protector))
#endif
__attribute__((noreturn)) void __exit(int status) {
  for (;;) {
    LIBC_NAMESPACE::syscall_impl<long>(__NR_exit_group, status);
    LIBC_NAMESPACE::syscall_impl<long>(__NR_exit, status);
  }
}

#if !defined(__APPLE__)

// Reference: libc/src/fcntl/linux/open.cpp
int __open(const char *path, int flags, mode_t mode) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_openat, AT_FDCWD, path, flags,
                                           mode);
}

// Reference: libc/src/unistd/linux/readlink.cpp
ssize_t __readlink(const char *path, char *buf, size_t bufsize) {
  return LIBC_NAMESPACE::syscall_impl<ssize_t>(__NR_readlinkat, AT_FDCWD, path,
                                               buf, bufsize);
}

// Reference: libc/src/unistd/linux/lseek.cpp
off_t __lseek(int fd, off_t offset, int whence) {
  return LIBC_NAMESPACE::syscall_impl<off_t>(__NR_lseek, fd, offset, whence);
}

// Reference: libc/src/unistd/linux/ftruncate.cpp
int __ftruncate(int fd, off_t len) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_ftruncate, fd, len);
}

// Reference: libc/src/unistd/linux/close.cpp
int __close(int fd) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_close, fd);
}

// Reference: libc/src/sys/utsname/linux/uname.cpp
int __uname(struct UtsNameTy *name) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_uname, name);
}

// Reference: libc/src/sys/mman/linux/mprotect.cpp
int __mprotect(void *start, size_t size, int prot) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_mprotect, start, size, prot);
}

// Reference: libc/src/signal/linux/kill.cpp
int __kill(pid_t pid, int sig) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_kill, pid, sig);
}

// Reference: libc/src/unistd/linux/fsync.cpp
int __fsync(int fd) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_fsync, fd);
}

// Reference: libc/src/sys/prctl/linux/prctl.cpp
int __prctl(int option, unsigned long arg2, unsigned long arg3,
            unsigned long arg4, unsigned long arg5) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_prctl, option, arg2, arg3, arg4,
                                           arg5);
}

// Reference: libc/src/sys/mman/linux/madvise.cpp
int __madvise(void *addr, size_t size, int advice) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_madvise, addr, size, advice);
}

// Reference: libc/src/time/linux/nanosleep.cpp
int __nanosleep(const timespec *req, timespec *rem) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_nanosleep, req, rem);
}

// FIXME: cleanup here
// Currently, the arch-specific implementations have not been changed in order
// to avoid breaking the instrumentation functionality:
//   - x64_64 - fork syscall
//   - aarch64/riscv64 - clone syscall
// arm64/riscv64 selects CONFIG_CLONE_BACKWARDS (child_tid 5th arg in sys_clone)
// but x86_64 doesn't (child_tid 4th arg).
// x86_64: https://github.com/torvalds/linux/blob/v7.1/kernel/fork.c#L2847
// arch64/riscv64:
// https://github.com/torvalds/linux/blob/v7.1/kernel/fork.c#L2831
// Not clear why CLONE_CHILD_CLEARTID | CLONE_CHILD_SETTID. In this case
// child_tid should be set.
// Reference (glibc): sysdeps/unix/sysv/linux/arch-fork.h
// Reference: libc/src/unistd/linux/fork.cpp
pid_t __fork() {
#if defined(__x86_64__)
  return LIBC_NAMESPACE::syscall_impl<pid_t>(__NR_fork);
#elif defined(__aarch64__) || defined(__arm64__) || defined(__riscv)
  return LIBC_NAMESPACE::syscall_impl<pid_t>(
      __NR_clone, CLONE_CHILD_CLEARTID | CLONE_CHILD_SETTID | SIGCHLD, 0, 0, 0,
      0);
#endif
}

// Reference: libc/src/__support/File/linux/dir.cpp
ssize_t __getdents64(unsigned int fd, dirent64 *dirp, unsigned int count) {
  return LIBC_NAMESPACE::syscall_impl<ssize_t>(__NR_getdents64, fd, dirp,
                                               count);
}

// Reference: libc/src/unistd/linux/getppid.cpp
pid_t __getppid() { return LIBC_NAMESPACE::syscall_impl<pid_t>(__NR_getppid); }

// No reference information was found in LLVM's libc; glibc and the Linux kernel
// were used instead.

// Reference
// - glibc: posix/getpgid.c
//      pid_t __getpgid (pid_t pid)
// - Linux kernel: kernel/sys.c
//      SYSCALL_DEFINE1(getpgid, pid_t, pid)
pid_t __getpgid(pid_t pid) {
  return LIBC_NAMESPACE::syscall_impl<pid_t>(__NR_getpgid, pid);
}

// Reference
// - glibc: posix/setpgid.c
//      int __setpgid (int pid, int pgid)
// - Linux kernel: kernel/sys.c
//      SYSCALL_DEFINE2(setpgid, pid_t, pid, pid_t, pgid)
int __setpgid(pid_t pid, pid_t pgid) {
  return LIBC_NAMESPACE::syscall_impl<int>(__NR_setpgid, pid, pgid);
}

#endif

} // anonymous namespace

#endif /* LLVM_TOOLS_LLVM_BOLT_SYSCALL_WRAPPERS */
