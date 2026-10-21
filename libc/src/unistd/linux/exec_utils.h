//===-- Linux implementation helpers for exec -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_LINUX_EXEC_UTILS_H
#define LLVM_LIBC_SRC_UNISTD_LINUX_EXEC_UTILS_H

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/execve.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/environ_internal.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// TODO: POSIX requires that if execve fails with ENOEXEC, the file is executed
// using the system shell (as if by `sh -- file args...`). The kernel only
// handles ELF, "#!" scripts, and binfmt_misc formats, so shebang-less scripts
// currently fail with ENOEXEC instead of being run by /bin/sh.
LIBC_INLINE int execvpe(const char *file, char *const argv[],
                        char *const envp[]) {
  cpp::string_view file_sv(file);
  if (file_sv.empty()) {
    libc_errno = ENOENT;
    return -1;
  }

  // If there's a / in file, treat it as a path.
  if (file_sv.find_first_of('/') != cpp::string_view::npos) {
    auto ret = linux_syscalls::execve(file, argv, envp);
    if (!ret) {
      libc_errno = ret.error();
      return -1;
    }
    return *ret;
  }

  // Otherwise treat file as the name of a file in a directory, search the
  // directories listed in PATH.
  constexpr cpp::string_view DEFAULT_PATH = "/bin:/usr/bin";
  const char *env_path =
      internal::EnvironmentManager::get_instance().get("PATH");
  cpp::string_view path_env =
      env_path != nullptr ? cpp::string_view(env_path) : DEFAULT_PATH;

  char buf[PATH_MAX];
  bool seen_eacces = false;
  int last_error = ENOENT;

  for (size_t colon_pos = 0; colon_pos != cpp::string_view::npos;
       path_env = path_env.substr(colon_pos + 1)) {
    colon_pos = path_env.find_first_of(':');
    cpp::string_view dir = path_env.substr(0, colon_pos);
    if (dir.empty())
      dir = ".";

    if (dir.size() + 1 + file_sv.size() >= PATH_MAX) {
      last_error = ENAMETOOLONG;
      continue;
    }

    inline_memcpy(buf, dir.data(), dir.size());
    buf[dir.size()] = '/';
    inline_memcpy(buf + dir.size() + 1, file_sv.data(), file_sv.size());
    buf[dir.size() + 1 + file_sv.size()] = '\0';

    auto ret = linux_syscalls::execve(buf, argv, envp);
    if (ret)
      return *ret;

    int err = ret.error();
    if (err == EACCES) {
      seen_eacces = true;
    } else if (err == ENOENT || err == ENOTDIR) {
      last_error = err;
    } else {
      libc_errno = err;
      return -1;
    }
  }

  libc_errno = seen_eacces ? EACCES : last_error;
  return -1;
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_LINUX_EXEC_UTILS_H
