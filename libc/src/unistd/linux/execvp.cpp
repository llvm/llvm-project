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
#include "hdr/limits_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/execve.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/stdlib/environ_internal.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/unistd/environ.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

LIBC_INLINE ErrorOr<int> execve_with_fallback(const char *path,
                                              char *const argv[],
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
  return ret;
}

} // namespace

LLVM_LIBC_FUNCTION(int, execvp, (const char *file, char *const argv[])) {
  LIBC_CRASH_ON_NULLPTR(file);
  LIBC_CRASH_ON_NULLPTR(argv);

  cpp::string_view file_sv(file);
  if (file_sv.empty()) {
    libc_errno = ENOENT;
    return -1;
  }

  char *const *envp = const_cast<char *const *>(LIBC_NAMESPACE::environ);

  // If there's a / in file, treat it as a path.
  if (file_sv.find_first_of('/') != cpp::string_view::npos) {
    auto ret = execve_with_fallback(file, argv, envp);
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

    auto ret = execve_with_fallback(buf, argv, envp);
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

} // namespace LIBC_NAMESPACE_DECL
