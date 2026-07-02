//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of tmpnam, a POSIX function that generate a string that is a
/// valid pathname that does not name an existing file.
/// See:
/// https://pubs.opengroup.org/onlinepubs/9799919799/functions/tmpnam.html
///
//===----------------------------------------------------------------------===//

#include "src/stdio/tmpnam.h"
#include "hdr/errno_macros.h"
#include "hdr/stdio_macros.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/access.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/macros/config.h"
#include "src/stdio/snprintf.h"

namespace LIBC_NAMESPACE_DECL {

static char tmpbuf[L_tmpnam];
static cpp::Atomic<size_t> tmpnam_budget = TMP_MAX;

/* partially thread-safe */
LLVM_LIBC_FUNCTION(char *, tmpnam, (char *s)) {
  if (s == nullptr)
    s = tmpbuf;

  // here if the s is null then use tmpbuf and if sizeof
  // POSIX portable filename character set, sorted by ASCII value.
  // See
  // https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap03.html#tag_03_265
  const char charset[] = "-._0123456789"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz";

  int prefix_size = LIBC_NAMESPACE::snprintf(s, L_tmpnam, "%s/", P_tmpdir);

  bool is_unique = false;
  while (!is_unique) {
    size_t curr_budget = tmpnam_budget.load(cpp::MemoryOrder::RELAXED);

    do {
      if (curr_budget == 0)
        break;
    } while (
        !tmpnam_budget.compare_exchange_strong(curr_budget, curr_budget - 1));

    if (curr_budget == 0)
      break;

    for (size_t i = prefix_size; i < L_tmpnam - 1; i++) {
      uint8_t rand_byte;
      auto ret = linux_syscalls::getrandom(&rand_byte, 1, 0);
      if (!ret.has_value()) {
        /* return nullptr when getrandom fails but consume tmpnam budget */
        return nullptr;
      }
      s[i] = charset[rand_byte % (sizeof(charset) - 1)];
    }
    s[L_tmpnam - 1] = '\0';
    auto res = linux_syscalls::access(s, F_OK);
    is_unique = res.has_value() ? false : true;
  }

  if (is_unique)
    return s;
  /* implementation-defined: if we exhaust budget we return nullptr */
  return nullptr;
}
} // namespace LIBC_NAMESPACE_DECL
