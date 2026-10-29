//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the POSIX popen function for Linux.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/popen.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/types/FILE.h"
#include "hdr/types/pid_t.h"
#include "hdr/unistd_macros.h"
#include "src/__support/File/file.h"
#include "src/__support/File/file_mode.h"
#include "src/__support/File/linux/file.h"
#include "src/__support/File/linux/popen_file.h"
#include "src/__support/OSUtil/linux/shell_fork.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/close.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/dup2.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/fcntl.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/pipe2.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

// A pipe only goes one way, so these are the only modes popen accepts.
static constexpr FileMode VALID_POPEN_MODES[] = {
    FileMode("r"), FileMode("w"), FileMode("re"), FileMode("we")};

// TODO: Simplify this with a cpp::find (or similar) helper in
// src/__support/CPP/algorithm.h once one exists.
static bool is_valid_popen_mode(const FileMode &file_mode) {
  for (const FileMode &valid_mode : VALID_POPEN_MODES)
    if (file_mode == valid_mode)
      return true;
  return false;
}

// Clears FD_CLOEXEC on |fd|, leaving its other descriptor flags unchanged.
static ErrorOr<int> clear_close_on_exec(int fd) {
  ErrorOr<int> flags = linux_syscalls::fcntl(fd, F_GETFD);
  if (!flags)
    return flags;
  return linux_syscalls::fcntl(
      fd, F_SETFD, reinterpret_cast<void *>(flags.value() & ~FD_CLOEXEC));
}

// Makes |fd| available as |target_fd| in a process that is about to exec.
// Both ends of the pipe are created close-on-exec, so if |fd| is already
// |target_fd| its FD_CLOEXEC flag has to be cleared instead.
static bool duplicate_fd_for_exec(int fd, int target_fd) {
  if (fd == target_fd)
    return clear_close_on_exec(fd).has_value();
  return linux_syscalls::dup2(fd, target_fd).has_value();
}

LLVM_LIBC_FUNCTION(::FILE *, popen, (const char *command, const char *mode)) {
  LIBC_CRASH_ON_NULLPTR(command);
  LIBC_CRASH_ON_NULLPTR(mode);

  const FileMode file_mode(mode);
  if (!is_valid_popen_mode(file_mode)) {
    libc_errno = EINVAL;
    return nullptr;
  }
  const bool read = file_mode.is_read();

  // Both ends start out close-on-exec so that no other child process can
  // inherit them. The child duplicates its end onto stdin or stdout before
  // exec.
  int pipe_fds[2];
  if (ErrorOr<int> res = linux_syscalls::pipe2(pipe_fds, O_CLOEXEC); !res) {
    libc_errno = res.error();
    return nullptr;
  }
  const int parent_fd = read ? pipe_fds[0] : pipe_fds[1];
  const int child_fd = read ? pipe_fds[1] : pipe_fds[0];
  const int child_target_fd = read ? STDOUT_FILENO : STDIN_FILENO;

  ErrorOr<PopenFile *> new_file =
      LinuxFile::create_buffered<PopenFile>(parent_fd, file_mode);
  if (!new_file) {
    linux_syscalls::close(parent_fd);
    linux_syscalls::close(child_fd);
    libc_errno = new_file.error();
    return nullptr;
  }
  PopenFile *file = new_file.value();

  // Hold the popen list lock across the fork so that the child closes exactly
  // the popen streams which are open in the parent.
  PopenFile::lock_popen_list();
  ErrorOr<pid_t> fork_result = shell_fork(command, [&] {
    PopenFile::close_popen_list_fds_unlocked();
    return duplicate_fd_for_exec(child_fd, child_target_fd);
  });
  linux_syscalls::close(child_fd);

  if (!fork_result) {
    PopenFile::unlock_popen_list();
    // The file is not on any list yet, so closing it just closes parent_fd
    // and frees it.
    file->close();
    libc_errno = fork_result.error();
    return nullptr;
  }

  file->set_pid(fork_result.value());
  if (!file_mode.is_close_on_exec()) {
    // This can only fail if parent_fd is invalid, which it is not.
    clear_close_on_exec(parent_fd);
  }
  PopenFile::add_to_popen_list_unlocked(file);
  PopenFile::unlock_popen_list();

  File::add_file(file);
  return reinterpret_cast<::FILE *>(file);
}

} // namespace LIBC_NAMESPACE_DECL
