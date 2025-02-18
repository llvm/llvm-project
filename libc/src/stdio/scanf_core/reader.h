//===-- Reader definition for scanf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H

#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/__support/macros/config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace scanf_core {

template <typename Derived> struct ReadBuffer {
  LIBC_INLINE char getc() { return static_cast<Derived *>(this)->getc(); }
  LIBC_INLINE void ungetc(int c) { static_cast<Derived *>(this)->ungetc(c); }
};

class StringBuffer : public ReadBuffer<StringBuffer> {
  const char *buffer;
  [[maybe_unused]] size_t buff_len;
  size_t buff_cur = 0;

public:
  LIBC_INLINE StringBuffer(const char *buffer, size_t buff_len)
      : buffer(buffer), buff_len(buff_len) {}

  LIBC_INLINE char getc() {
    char output = buffer[buff_cur];
    ++buff_cur;
    return output;
  }
  LIBC_INLINE void ungetc(int) {
    if (buff_cur > 0) {
      // While technically c should be written back to the buffer, in scanf we
      // always write the character that was already there. Additionally, the
      // buffer is most likely to contain a string that isn't part of a file,
      // which may not be writable.
      --buff_cur;
    }
  }
};

// TODO: We should be able to fold ReadBuffer into Reader.
template <typename T> class Reader {
  ReadBuffer<T> *buffer;
  size_t cur_chars_read = 0;

public:
  LIBC_INLINE Reader(ReadBuffer<T> *buffer) : buffer(buffer) {}

  // This returns the next character from the input and advances it by one
  // character. When it hits the end of the string or file it returns '\0' to
  // signal to stop parsing.
  LIBC_INLINE char getc() {
    ++cur_chars_read;
    return buffer->getc();
  }

  // This moves the input back by one character, placing c into the buffer if
  // this is a file reader, else c is ignored.
  LIBC_INLINE void ungetc(char c) {
    --cur_chars_read;
    buffer->ungetc(c);
  }

  LIBC_INLINE size_t chars_read() { return cur_chars_read; }
};

} // namespace scanf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
