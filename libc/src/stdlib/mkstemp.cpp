//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of mkstemp, a POSIX function that creates a unique temporary
/// file from a template string ending in at least six 'X' characters.
///
/// Replaces the trailing X's with random characters from the POSIX portable
/// filename character set, opens the file exclusively, and returns an open
/// file descriptor, retrying automatically on name collision. See:
/// https://pubs.opengroup.org/onlinepubs/9799919799/functions/mkdtemp.html
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/mkstemp.h"
#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/open.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, mkstemp, (char *tmpl)) {
  LIBC_CRASH_ON_NULLPTR(tmpl);

  cpp::string_view str_view(tmpl);
  size_t count = 0;
  size_t len = str_view.size();

  for (size_t i = len; i > 0; i--) {
    if (str_view[i - 1] != 'X')
      break;
    count++;
  }

  if (count < 6) {
    libc_errno = EINVAL;
    return -1;
  }

  char *suffix = tmpl + len - count;

  // POSIX portable filename character set, sorted by ASCII value.
  // See
  // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap03.html#tag_03_265
  const char charset[] = "-._0123456789"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz";

  int result = -1;
  bool file_created = false;
  while (!file_created) {
    for (size_t i = 0; i < count; i++) {
      uint8_t rand_byte;
      auto ret = linux_syscalls::getrandom(&rand_byte, 1, 0);
      if (!ret.has_value()) {
        libc_errno = ret.error();
        return -1;
      }
      // sizeof(charset) - 1 to account for the null terminator
      suffix[i] = charset[rand_byte % (sizeof(charset) - 1)];
    }

    auto fd = linux_syscalls::open(tmpl, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (!fd.has_value()) {
      if (fd.error() == EEXIST)
        continue;
      libc_errno = fd.error();
      return -1;
    }
    result = fd.value();
    file_created = true;
  }
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
