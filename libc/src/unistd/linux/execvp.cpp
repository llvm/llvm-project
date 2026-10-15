//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of execvp.
///
//===----------------------------------------------------------------------===//

#include "src/unistd/execvp.h"

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/execve.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/unistd/environ.h"

#include <linux/limits.h>

namespace LIBC_NAMESPACE_DECL {

namespace {

LIBC_INLINE const char *get_path() {
  if (LIBC_NAMESPACE::environ == nullptr)
    return nullptr;

  for (char **env = LIBC_NAMESPACE::environ; *env != nullptr; ++env) {
    cpp::string_view entry(*env);
    if (entry.starts_with("PATH="))
      return *env + 5;
  }
  return nullptr;
}

LIBC_INLINE int execve_with_fallback(const char *path, char *const argv[],
                                     char *const envp[]) {
  auto ret = linux_syscalls::execve(path, argv, envp);
  if (!ret && ret.error() == ENOEXEC) {
    size_t argc = 0;
    while (argv[argc] != nullptr)
      ++argc;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"
#if __has_warning("-Wvla-cxx-extension")
#pragma GCC diagnostic ignored "-Wvla-cxx-extension"
#endif
    char *new_argv[argc + 2];
#pragma GCC diagnostic pop
    new_argv[0] = const_cast<char *>("sh");
    new_argv[1] = const_cast<char *>(path);
    for (size_t i = 1; i <= argc; ++i)
      new_argv[i + 1] = argv[i];

    ret = linux_syscalls::execve("/bin/sh", new_argv, envp);
  }
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return *ret;
}

} // namespace

LLVM_LIBC_FUNCTION(int, execvp, (const char *file, char *const argv[])) {
  LIBC_CRASH_ON_NULLPTR(file);

  if (file[0] == '\0') {
    libc_errno = ENOENT;
    return -1;
  }

  char *const *envp = const_cast<char *const *>(LIBC_NAMESPACE::environ);

  cpp::string_view file_view(file);
  if (file_view.contains('/'))
    return execve_with_fallback(file, argv, envp);

  const char *path = get_path();
  // POSIX standard default path if PATH is not set.
  if (path == nullptr)
    path = ":/bin:/usr/bin";

  cpp::string_view path_view(path);
  bool seen_eacces = false;

  size_t start = 0;
  while (start <= path_view.size()) {
    size_t end = path_view.find_first_of(':', start);
    if (end == cpp::string_view::npos)
      end = path_view.size();

    cpp::string_view dir = path_view.substr(start, end - start);
    start = end + 1;

    char path_buf[PATH_MAX];
    cpp::StringStream ss(path_buf);

    if (dir.empty()) {
      ss << file << cpp::StringStream::ENDS;
    } else {
      ss << dir;
      if (!dir.ends_with('/'))
        ss << '/';
      ss << file << cpp::StringStream::ENDS;
    }

    if (ss.overflow())
      continue;

    auto ret = linux_syscalls::execve(path_buf, argv, envp);
    if (!ret && ret.error() == ENOEXEC) {
      size_t argc = 0;
      while (argv[argc] != nullptr)
        ++argc;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"
#if __has_warning("-Wvla-cxx-extension")
#pragma GCC diagnostic ignored "-Wvla-cxx-extension"
#endif
      char *new_argv[argc + 2];
#pragma GCC diagnostic pop
      new_argv[0] = const_cast<char *>("sh");
      new_argv[1] = path_buf;
      for (size_t i = 1; i <= argc; ++i)
        new_argv[i + 1] = argv[i];

      ret = linux_syscalls::execve("/bin/sh", new_argv, envp);
    }

    if (ret)
      return *ret;

    int err = ret.error();
    if (err == EACCES)
      seen_eacces = true;
    else if (err != ENOENT && err != ENOTDIR) {
      libc_errno = err;
      return -1;
    }

    if (end == path_view.size())
      break;
  }

  libc_errno = seen_eacces ? EACCES : ENOENT;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
