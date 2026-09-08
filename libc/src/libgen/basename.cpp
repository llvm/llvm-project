//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of basename.
///
//===----------------------------------------------------------------------===//

#include "src/libgen/basename.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, basename, (char *path)) {
  if (path == nullptr || path[0] == '\0')
    return const_cast<char *>(".");

  cpp::string_view sv(path);
  size_t last_non_slash = sv.find_last_not_of('/');

  if (last_non_slash == cpp::string_view::npos)
    return const_cast<char *>("/");

  size_t last_slash = sv.substr(0, last_non_slash).find_last_of('/');

  size_t start = (last_slash == cpp::string_view::npos) ? 0 : last_slash + 1;
  size_t end = last_non_slash + 1;

  if (end < sv.size())
    path[end] = '\0';

  return path + start;
}

} // namespace LIBC_NAMESPACE_DECL
