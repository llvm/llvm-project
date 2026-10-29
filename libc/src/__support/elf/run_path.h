//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Expanding a run path entry
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_RUN_PATH_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_RUN_PATH_H

#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/readlink.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

LIBC_INLINE bool starts_with(const char *text, const char *prefix) {
  for (; *prefix != '\0'; ++text, ++prefix)
    if (*text != *prefix)
      return false;
  return true;
}

// How much of a path names the directory holding it, with the separator
// dropped. Zero when there is no directory part at all.
LIBC_INLINE size_t directory_length(const char *path) {
  size_t last = 0;
  for (size_t i = 0; path[i] != '\0'; ++i)
    if (path[i] == '/')
      last = i;
  // A path rooted at / has the slash itself as its directory.
  if (last == 0 && path[0] == '/')
    return 1;
  return last;
}

// The longest path the executable is resolved to. A longer one keeps the path
// the program was started by.
constexpr size_t MAX_EXECUTABLE_PATH = 4096;

// What $ORIGIN stands for in the executable's own run paths: the directory of
// the file the program really is. A command is often started through a link,
// such as a versioned tool's name under /usr/bin, and its libraries sit beside
// the file the link leads to, which is where glibc and musl look as well. The
// kernel says which file that is through /proc. Where there is no /proc to ask,
// as early in boot, or the path does not fit, the path the program was started
// by is kept.
LIBC_INLINE const char *executable_origin(const char *started_as, char *buffer,
                                          size_t capacity) {
  if (capacity < 2)
    return started_as;
  auto length = linux_syscalls::readlink("/proc/self/exe", buffer, capacity);
  if (!length.has_value() || length.value() <= 0 ||
      static_cast<size_t>(length.value()) >= capacity)
    return started_as;
  buffer[length.value()] = '\0';
  return buffer;
}

// Copies one run path entry into `out`, replacing $ORIGIN with the directory
// holding the object whose run path it is. Both spellings the dynamic linkers
// accept are handled. Returns false if it does not fit, if the entry is
// empty, or if it needs an origin and there is none to give it.
LIBC_INLINE bool expand_run_path(const char *entry, size_t length,
                                 const char *origin, char *out,
                                 size_t capacity) {
  if (capacity == 0)
    return false;
  size_t n = 0;
  for (size_t i = 0; i < length;) {
    if (entry[i] == '$') {
      const char *rest = entry + i + 1;
      const size_t remaining = length - i - 1;
      size_t token = 0;
      if (remaining >= 6 && starts_with(rest, "ORIGIN"))
        token = 7;
      else if (remaining >= 8 && starts_with(rest, "{ORIGIN}"))
        token = 9;
      if (token != 0) {
        if (origin == nullptr)
          return false;
        const size_t directory = directory_length(origin);
        if (directory == 0) {
          // The object was named without a directory, so it came from the
          // working directory and that is what $ORIGIN means here.
          if (n + 1 >= capacity)
            return false;
          out[n++] = '.';
        } else {
          if (n + directory >= capacity)
            return false;
          for (size_t j = 0; j < directory; ++j)
            out[n++] = origin[j];
        }
        i += token;
        continue;
      }
    }
    if (n + 1 >= capacity)
      return false;
    out[n++] = entry[i++];
  }
  out[n] = '\0';
  return n != 0;
}

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_RUN_PATH_H
