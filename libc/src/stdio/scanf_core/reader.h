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
#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace scanf_core {

using StreamGetc = int (*)(void *);
using StreamUngetc = void (*)(int, void *);

// This is intended to be either a raw string or a buffer syncronized with the
// file's internal buffer.
struct ReadBuffer {
  char *buffer;
  size_t buff_len;
  size_t buff_cur = 0;
};

class Reader {
  ReadBuffer *rb;

  void *input_stream = nullptr;

  StreamGetc stream_getc = nullptr;
  StreamUngetc stream_ungetc = nullptr;

  size_t cur_chars_read = 0;

public:
  // TODO: Set buff_len with a proper constant
  LIBC_INLINE Reader(ReadBuffer *string_buffer) : rb(string_buffer) {}

  LIBC_INLINE Reader(void *stream, StreamGetc stream_getc_in,
                     StreamUngetc stream_ungetc_in,
                     ReadBuffer *stream_buffer = nullptr)
      : rb(stream_buffer), input_stream(stream), stream_getc(stream_getc_in),
        stream_ungetc(stream_ungetc_in) {}

  // This returns the next character from the input and advances it by one
  // character. When it hits the end of the string or file it returns '\0' to
  // signal to stop parsing.
  LIBC_INLINE char getc() {
    ++cur_chars_read;
    if (rb != nullptr) {
      char output = rb->buffer[rb->buff_cur];
      ++(rb->buff_cur);
      return output;
    }
    // This should reset the buffer if applicable.
    return static_cast<char>(stream_getc(input_stream));
  }

  // This moves the input back by one character, placing c into the buffer if
  // this is a file reader, else c is ignored.
  void ungetc(char c);

  LIBC_INLINE size_t chars_read() { return cur_chars_read; }
};

} // namespace scanf_core
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
