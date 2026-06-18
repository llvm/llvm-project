//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of dirname.
///
//===----------------------------------------------------------------------===//

#include "src/libgen/dirname.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, dirname, (char *path)) {
  if (path == nullptr || path[0] == '\0')
    return const_cast<char *>(".");

  cpp::string_view sv(path);
  size_t last_non_slash = sv.find_last_not_of('/');

  if (last_non_slash == cpp::string_view::npos)
    return const_cast<char *>("/");

  size_t last_slash = sv.substr(0, last_non_slash).find_last_of('/');

  if (last_slash == cpp::string_view::npos)
    return const_cast<char *>(".");

  cpp::string_view dir_sv = sv.substr(0, last_slash);
  size_t dir_last_non_slash = dir_sv.find_last_not_of('/');

  if (dir_last_non_slash == cpp::string_view::npos) {
    path[1] = '\0';
    return path;
  }

  path[dir_last_non_slash + 1] = '\0';
  return path;
}

} // namespace LIBC_NAMESPACE_DECL
