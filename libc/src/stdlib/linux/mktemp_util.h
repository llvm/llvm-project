//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared helper functions for creating unique temporary files and directories.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDLIB_LINUX_MKTEMP_UTIL_H
#define LLVM_LIBC_SRC_STDLIB_LINUX_MKTEMP_UTIL_H

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/getrandom.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// POSIX portable filename character set, sorted by ASCII value.
// See
// https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap03.html#tag_03_265
LIBC_INLINE_VAR constexpr cpp::string_view MKTEMP_CHARSET =
    "-._0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

// Minimum number of trailing 'X' characters required by POSIX.
LIBC_INLINE_VAR constexpr size_t MIN_MKTEMP_SUFFIX = 6;

/// Core helper function for creating temporary files and directories.
///
/// \param tmpl Template string ending in at least six 'X' characters.
/// \param create_fn Callable taking `const char *path` and returning
/// `ErrorOr<int>`.
/// \return `ErrorOr<int>` with the result of `create_fn` on success, or Error
/// on failure.
template <typename CreateFn>
LIBC_INLINE ErrorOr<int> mktemp_core(char *tmpl, CreateFn create_fn) {
  cpp::string_view str_view(tmpl);
  size_t len = str_view.size();
  if (len < MIN_MKTEMP_SUFFIX)
    return Error(EINVAL);

  size_t pos = str_view.find_last_not_of('X');
  size_t count = (pos == cpp::string_view::npos) ? len : len - pos - 1;

  if (count < MIN_MKTEMP_SUFFIX)
    return Error(EINVAL);

  cpp::span<char> suffix(tmpl + (len - count), count);

  // Maximum collision retry attempts before returning EEXIST per POSIX.
  constexpr size_t MAX_ATTEMPTS = 10000;
  // Read random bytes in batches to minimize getrandom syscall overhead.
  constexpr size_t BATCH_SIZE = 64;
  cpp::array<uint8_t, BATCH_SIZE> rand_buf;

  for (size_t attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
    for (size_t offset = 0; offset < count;) {
      size_t chunk =
          (count - offset < BATCH_SIZE) ? (count - offset) : BATCH_SIZE;
      auto ret = linux_syscalls::getrandom(rand_buf.data(), chunk, 0);
      if (!ret.has_value())
        return Error(ret.error());
      if (ret.value() == 0)
        return Error(EIO);
      for (size_t j = 0; j < static_cast<size_t>(ret.value()); ++j) {
        suffix[offset + j] =
            MKTEMP_CHARSET[rand_buf[j] % MKTEMP_CHARSET.size()];
      }
      offset += static_cast<size_t>(ret.value());
    }

    auto result = create_fn(tmpl);
    if (!result.has_value() && result.error() == EEXIST)
      continue;
    return result;
  }
  return Error(EEXIST);
}

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDLIB_LINUX_MKTEMP_UTIL_H
