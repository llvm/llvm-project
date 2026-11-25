//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation helpers for exec.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_UNISTD_LINUX_EXEC_UTILS_H
#define LLVM_LIBC_SRC_UNISTD_LINUX_EXEC_UTILS_H

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/execve.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/environ_internal.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Extracts argv from a variable argument list and calls exec(argv).
//
// The varargs list in `*args` is consumed up to and including the terminating
// null pointer. If additional arguments follow (such as envp in execle),
// they can be read from `*args` by the `exec` callback.
template <typename ExecFn>
LIBC_INLINE ErrorOr<int> exec_with_arg_list(const char *arg0, va_list *args,
                                            ExecFn exec) {
  va_list args_count;
  va_copy(args_count, *args);
  size_t argc = 1;
  while (va_arg(args_count, const char *) != nullptr)
    ++argc;
  va_end(args_count);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"
#if __has_warning("-Wvla-cxx-extension")
#pragma GCC diagnostic ignored "-Wvla-cxx-extension"
#endif
  char *argv[argc + 1];
#pragma GCC diagnostic pop
  argv[0] = const_cast<char *>(arg0);

  for (size_t i = 1; i < argc; ++i)
    argv[i] = va_arg(*args, char *);
  // Consume the terminating nullptr from *args.
  va_arg(*args, char *);
  argv[argc] = nullptr;

  return exec(argv);
}

LIBC_INLINE ErrorOr<int> execve_shell_fallback(const char *path,
                                               char *const argv[],
                                               char *const envp[]) {
  size_t argc = 0;
  if (argv != nullptr) {
    while (argv[argc] != nullptr)
      ++argc;
  }
  size_t count = (argc > 0) ? argc : 1;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"
#if __has_warning("-Wvla-cxx-extension")
#pragma GCC diagnostic ignored "-Wvla-cxx-extension"
#endif
  char *new_argv[count + 2];
#pragma GCC diagnostic pop
  new_argv[0] = const_cast<char *>("sh");
  new_argv[1] = const_cast<char *>(path);
  if (argv != nullptr && argc > 1) {
    for (size_t i = 1; i < argc; ++i)
      new_argv[i + 1] = argv[i];
  }
  new_argv[count + 1] = nullptr;

  constexpr const char *DEFAULT_BSHELL = "/bin/sh";
  return linux_syscalls::execve(DEFAULT_BSHELL, new_argv, envp);
}

LIBC_INLINE ErrorOr<int> execvpe(const char *file, char *const argv[],
                                 char *const envp[]) {
  cpp::string_view file_view(file);
  if (file_view.empty())
    return Error(ENOENT);

  // If there's a / in file, treat it as a path.
  if (file_view.find_first_of('/') != cpp::string_view::npos) {
    auto ret = linux_syscalls::execve(file, argv, envp);
    if (ret || ret.error() != ENOEXEC)
      return ret;
    return execve_shell_fallback(file, argv, envp);
  }

  // Otherwise treat file as the name of a file in a directory, search the
  // directories listed in PATH.
  //
  // Per POSIX.1-2024 and Linux / GNU specifications, the search path is taken
  // from the caller's environment (via EnvironmentManager), not from the `envp`
  // argument passed to execvpe.
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

    if (dir.size() + 1 + file_view.size() >= PATH_MAX) {
      last_error = ENAMETOOLONG;
      continue;
    }

    inline_memcpy(buf, dir.data(), dir.size());
    buf[dir.size()] = '/';
    inline_memcpy(buf + dir.size() + 1, file_view.data(), file_view.size());
    buf[dir.size() + 1 + file_view.size()] = '\0';

    auto ret = linux_syscalls::execve(buf, argv, envp);
    if (ret)
      return ret;

    switch (int err = ret.error()) {
    case EACCES:
      seen_eacces = true;
      break;
    case ENOENT:
    case ENOTDIR:
      last_error = err;
      break;
    case ENOEXEC:
      return execve_shell_fallback(buf, argv, envp);
    default:
      return Error(err);
    }
  }

  return Error(seen_eacces ? EACCES : last_error);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_UNISTD_LINUX_EXEC_UTILS_H
