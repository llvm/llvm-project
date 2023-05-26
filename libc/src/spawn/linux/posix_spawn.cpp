//===-- Linux implementation of posix_spawn -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/spawn/posix_spawn.h"

#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"
#include "src/spawn/file_actions.h"

#include <fcntl.h>
#include <signal.h> // For SIGCHLD
#include <spawn.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

namespace {

pid_t fork() {
  // TODO: Use only the clone syscall and use a sperate small stack in the child
  // to avoid duplicating the complete stack from the parent. A new stack will
  // be created on exec anyway so duplicating the full stack is unnecessary.
#ifdef SYS_fork
  return __llvm_libc::syscall_impl(SYS_fork);
#elif defined(SYS_clone)
  return __llvm_libc::syscall_impl(SYS_clone, SIGCHLD, 0);
#else
#error "fork or clone syscalls not available."
#endif
}

cpp::optional<int> open(const char *path, int oflags, mode_t mode) {
#ifdef SYS_open
  int fd = __llvm_libc::syscall_impl(SYS_open, path, oflags, mode);
#else
  int fd = __llvm_libc::syscall_impl(SYS_openat, AT_FDCWD, path, oflags, mode);
#endif
  if (fd > 0)
    return fd;
  // The open function is called as part of the child process' preparatory
  // steps. If an open fails, the child process just exits. So, unlike
  // the public open function, we do not need to set errno here.
  return cpp::nullopt;
}

void close(int fd) { __llvm_libc::syscall_impl(SYS_close, fd); }

// We use dup3 if dup2 is not available, similar to our implementation of dup2
bool dup2(int fd, int newfd) {
#ifdef SYS_dup2
  long ret = __llvm_libc::syscall_impl(SYS_dup2, fd, newfd);
#elif defined(SYS_dup3)
  long ret = __llvm_libc::syscall_impl(SYS_dup3, fd, newfd, 0);
#else
#error "dup2 and dup3 syscalls not available."
#endif
  return ret < 0 ? false : true;
}

// All exits from child_process are error exits. So, we use a simple
// exit implementation which exits with code 127.
void exit() {
  for (;;) {
    __llvm_libc::syscall_impl(SYS_exit_group, 127);
    __llvm_libc::syscall_impl(SYS_exit, 127);
  }
}

void child_process(const char *__restrict path,
                   const posix_spawn_file_actions_t *file_actions,
                   const posix_spawnattr_t *__restrict, // For now unused
                   char *const *__restrict argv, char *const *__restrict envp) {
  // TODO: In the code below, the child_process just exits on error during
  // processing |file_actions| and |attr|. The correct way would be to exit
  // after conveying the information about the failure to the parent process
  // (via a pipe for example).
  // TODO: Handle |attr|.

  if (file_actions != nullptr) {
    auto *act = reinterpret_cast<BaseSpawnFileAction *>(file_actions->__front);
    while (act != nullptr) {
      switch (act->type) {
      case BaseSpawnFileAction::OPEN: {
        auto *open_act = reinterpret_cast<SpawnFileOpenAction *>(act);
        auto fd = open(open_act->path, open_act->oflag, open_act->mode);
        if (!fd)
          exit();
        int actual_fd = *fd;
        if (actual_fd != open_act->fd) {
          bool dup2_result = dup2(actual_fd, open_act->fd);
          close(actual_fd); // The old fd is not needed anymore.
          if (!dup2_result)
            exit();
        }
        break;
      }
      case BaseSpawnFileAction::CLOSE: {
        auto *close_act = reinterpret_cast<SpawnFileCloseAction *>(act);
        close(close_act->fd);
        break;
      }
      case BaseSpawnFileAction::DUP2: {
        auto *dup2_act = reinterpret_cast<SpawnFileDup2Action *>(act);
        if (!dup2(dup2_act->fd, dup2_act->newfd))
          exit();
        break;
      }
      }
      act = act->next;
    }
  }

  if (__llvm_libc::syscall_impl(SYS_execve, path, argv, envp) < 0)
    exit();
}

} // anonymous namespace

LLVM_LIBC_FUNCTION(int, posix_spawn,
                   (pid_t *__restrict pid, const char *__restrict path,
                    const posix_spawn_file_actions_t *file_actions,
                    const posix_spawnattr_t *__restrict attr,
                    char *const *__restrict argv,
                    char *const *__restrict envp)) {
  pid_t cpid = fork();
  if (cpid == 0)
    child_process(path, file_actions, attr, argv, envp);
  else if (cpid < 0)
    return -cpid;

  if (pid != nullptr)
    *pid = cpid;

  // TODO: Before returning, one should wait for the child_process to startup
  // successfully. For now, we will just return. Future changes will add proper
  // wait (using pipes for example).

  return 0;
}

} // namespace __llvm_libc
