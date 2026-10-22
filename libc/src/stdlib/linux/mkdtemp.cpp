//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of mkdtemp, a POSIX function that creates a unique temporary
/// directory from a template string ending in at least six 'X' characters.
///
/// Replaces the trailing X's with random characters from the POSIX portable
/// filename character set, creates the directory with 0700 permissions,
/// and returns the pathname, retrying automatically on name collision. See:
/// https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/mkdtemp.h"
#include "hdr/sys_stat_macros.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/mkdir.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/stdlib/linux/mktemp_util.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(char *, mkdtemp, (char *tmpl)) {
  LIBC_CRASH_ON_NULLPTR(tmpl);

  auto res = internal::mktemp_core(tmpl, [](const char *path) {
    return linux_syscalls::mkdir(path, S_IRWXU);
  });
  if (!res.has_value()) {
    libc_errno = res.error();
    return nullptr;
  }
  return tmpl;
}

} // namespace LIBC_NAMESPACE_DECL
