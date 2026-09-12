//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of tmpnam.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/tmpnam.h"
#include "hdr/errno_macros.h"  // For ENOENT
#include "hdr/stdio_macros.h"  // For L_tmpnam and TMP_MAX
#include "hdr/unistd_macros.h" // For F_OK
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/access.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

static char TMPBUF[L_tmpnam];
static cpp::Atomic<size_t> tmpnam_budget = TMP_MAX;

// Thread-safety guarantees per ISO C & POSIX:
// - When 's' is nullptr, callers share internal static storage ('TMPBUF'),
// which is not thread-safe per standard specification allowance.
// - Process-wide call budget ('tmpnam_budget') is tracked atomically using a
// lock-free compare-and-swap loop.
LLVM_LIBC_FUNCTION(char *, tmpnam, (char *s)) {
  if (s == nullptr)
    s = TMPBUF;

  // Subset of the POSIX portable filename character set.
  // Deliberately sized to 64 (a power of 2, rather than the full set)
  // to prevent slight modulo bias also helps in performance.
  const char CHARSET[] = "0123456789"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                         "abcdefghijklmnopqrstuvwxyz"
                         "_.";

  constexpr size_t CHARSET_SIZE = sizeof(CHARSET) - 1;
  static_assert((CHARSET_SIZE & (CHARSET_SIZE - 1)) == 0,
                "CHARSET_SIZE must be a power of 2");

  constexpr size_t MASK = CHARSET_SIZE - 1;

  constexpr cpp::string_view PREFIX = "/tmp/";
  constexpr size_t PREFIX_SIZE = PREFIX.size();

  static_assert(PREFIX_SIZE + 1 < L_tmpnam, "L_tmpnam is too small");

  inline_memcpy(s, PREFIX.data(), PREFIX_SIZE);
  constexpr size_t SUFFIX_SIZE = L_tmpnam - PREFIX_SIZE - 1;

  while (true) {
    size_t curr_budget = tmpnam_budget.load(cpp::MemoryOrder::RELAXED);

    do {
      if (curr_budget == 0)
        break;
    } while (
        !tmpnam_budget.compare_exchange_strong(curr_budget, curr_budget - 1));

    if (curr_budget == 0)
      break;

    uint8_t rand_bytes[L_tmpnam];
    auto ret = linux_syscalls::getrandom(rand_bytes, SUFFIX_SIZE, 0);
    if (!ret.has_value() || ret.value() != SUFFIX_SIZE) {
      // return nullptr when getrandom fails but consume tmpnam budget
      return nullptr;
    }

    char *suffix = s + PREFIX_SIZE;
    for (size_t i = 0; i < SUFFIX_SIZE; i++) {
      suffix[i] = CHARSET[rand_bytes[i] & MASK];
    }

    s[L_tmpnam - 1] = '\0';
    auto res = linux_syscalls::access(s, F_OK);
    if (!res.has_value() && res.error() == ENOENT)
      return s;
  }

  // implementation-defined: if we exhaust budget we return nullptr
  return nullptr;
}
} // namespace LIBC_NAMESPACE_DECL
