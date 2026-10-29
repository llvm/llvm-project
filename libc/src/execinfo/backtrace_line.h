//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Rendering one line of a backtrace
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_EXECINFO_BACKTRACE_LINE_H
#define LLVM_LIBC_SRC_EXECINFO_BACKTRACE_LINE_H

#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace execinfo {

// The most a line can take: two paths' worth of names, the offset and the
// address. A name longer than this is cut rather than allowed to run over.
constexpr size_t MAX_LINE = 1024;

// Writes `text` and returns where the next thing goes, without ever passing
// `end`.
LIBC_INLINE char *put(char *out, char *end, const char *text) {
  if (text == nullptr)
    return out;
  for (; *text != '\0' && out < end; ++text)
    *out++ = *text;
  return out;
}

LIBC_INLINE char *put(char *out, char *end, char c) {
  if (out < end)
    *out++ = c;
  return out;
}

// Writes `value` in hexadecimal, with the 0x on the front that a backtrace
// line carries.
LIBC_INLINE char *put_hex(char *out, char *end, uintptr_t value) {
  out = put(out, end, "0x");
  char digits[2 * sizeof(uintptr_t)];
  size_t count = 0;
  do {
    digits[count++] = "0123456789abcdef"[value % 16];
    value /= 16;
  } while (value != 0);
  while (count > 0)
    out = put(out, end, digits[--count]);
  return out;
}

} // namespace execinfo
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_EXECINFO_BACKTRACE_LINE_H
