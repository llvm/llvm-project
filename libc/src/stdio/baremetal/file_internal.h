//===--- Helper functions for file I/O on baremetal -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_BAREMETAL_FILE_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_BAREMETAL_FILE_INTERNAL_H

#include "hdr/types/FILE.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {

// TODO: Deduplicate this with __support/File/file.h.
struct FileIOResult {
  size_t value;
  int error;

  constexpr FileIOResult(size_t val) : value(val), error(0) {}
  constexpr FileIOResult(size_t val, int error) : value(val), error(error) {}

  constexpr bool has_error() { return error != 0; }

  constexpr operator size_t() { return value; }
};

// ungetc handling.
int push_ungetc_value(::FILE *stream, int c);
bool pop_ungetc_value(::FILE *stream, unsigned char &out);

LIBC_INLINE int ungetc_internal(int c, ::FILE *stream) {
  return push_ungetc_value(stream, c);
}

LIBC_INLINE FileIOResult read_internal(char *buf, size_t size, ::FILE *stream) {
  if (size == 0)
    return 0;

  unsigned char ungetc_value = 0;
  size_t ungetc_value_copied = 0;

  if (pop_ungetc_value(stream, ungetc_value)) {
    buf[0] = static_cast<char>(ungetc_value);
    ungetc_value_copied = 1;

    if (size == 1)
      return 1;
  }

  ssize_t ret = __llvm_libc_stdio_read(stream, buf + ungetc_value_copied,
                                       size - ungetc_value_copied);
  if (ret < 0)
    return {ungetc_value_copied, static_cast<int>(-ret)};

  return ret + ungetc_value_copied;
}

LIBC_INLINE FileIOResult write_internal(const char *buf, size_t size,
                                        ::FILE *stream) {
  ssize_t ret = __llvm_libc_stdio_write(stream, buf, size);
  if (ret < 0)
    return {0, static_cast<int>(-ret)};
  return ret;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_BAREMETAL_FILE_INTERNAL_H
