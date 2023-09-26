//===-- Writer definition for printf ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
#define LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H

#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/optimization.h"
#include "src/stdio/printf_core/core_structs.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "src/string/memory_utils/inline_memset.h"

#include <stddef.h>

namespace LIBC_NAMESPACE {
namespace printf_core {

struct WriteBuffer {
  using StreamWriter = int (*)(cpp::string_view, void *);
  char *buff;
  const size_t buff_len;
  size_t buff_cur = 0;

  // The stream writer will be called when the buffer is full. It will be passed
  // string_views to write to the stream.
  StreamWriter stream_writer;
  void *output_target;

  LIBC_INLINE WriteBuffer(char *Buff, size_t Buff_len, StreamWriter hook,
                          void *target)
      : buff(Buff), buff_len(Buff_len), stream_writer(hook),
        output_target(target) {}

  LIBC_INLINE WriteBuffer(char *Buff, size_t Buff_len)
      : buff(Buff), buff_len(Buff_len), stream_writer(nullptr),
        output_target(nullptr) {}

  // The overflow_write method is intended to be called to write the contents of
  // the buffer and new_str to the stream_writer if it exists, else it will
  // write as much of new_str to the buffer as it can. The current position in
  // the buffer will be reset iff stream_writer is called. Calling this with an
  // empty string will flush the buffer if relevant.
  int overflow_write(cpp::string_view new_str) {
    // If there is a stream_writer, write the contents of the buffer, then
    // new_str, then clear the buffer.
    if (stream_writer != nullptr) {
      if (buff_cur > 0) {
        int retval = stream_writer({buff, buff_cur}, output_target);
        if (retval < 0) {
          return retval;
        }
      }
      if (new_str.size() > 0) {
        int retval = stream_writer(new_str, output_target);
        if (retval < 0) {
          return retval;
        }
      }
      buff_cur = 0;
      return WRITE_OK;
    } else {
      // We can't flush to the stream, so fill the rest of the buffer, then drop
      // the overflow.
      if (buff_cur < buff_len) {
        size_t bytes_to_write = buff_len - buff_cur;
        if (bytes_to_write > new_str.size()) {
          bytes_to_write = new_str.size();
        }
        inline_memcpy(buff + buff_cur, new_str.data(), bytes_to_write);
        buff_cur += bytes_to_write;
      }
      return WRITE_OK;
    }
  }
};

class Writer final {
  WriteBuffer *wb;
  int chars_written = 0;

  // This is a separate, non-inlined function so that the inlined part of the
  // write function is shorter.
  int pad(char new_char, size_t length);

public:
  LIBC_INLINE Writer(WriteBuffer *WB) : wb(WB) {}

  // Takes a string, copies it into the buffer if there is space, else passes it
  // to the overflow mechanism to be handled separately.
  LIBC_INLINE int write(cpp::string_view new_string) {
    chars_written += new_string.size();
    if (LIBC_LIKELY(wb->buff_cur + new_string.size() <= wb->buff_len)) {
      inline_memcpy(wb->buff + wb->buff_cur, new_string.data(),
                    new_string.size());
      wb->buff_cur += new_string.size();
      return WRITE_OK;
    }
    return wb->overflow_write(new_string);
  }

  // Takes a char and a length, memsets the next length characters of the buffer
  // if there is space, else calls pad which will loop and call the overflow
  // mechanism on a secondary buffer.
  LIBC_INLINE int write(char new_char, size_t length) {
    chars_written += length;

    if (LIBC_LIKELY(wb->buff_cur + length <= wb->buff_len)) {
      inline_memset(wb->buff + wb->buff_cur, new_char, length);
      wb->buff_cur += length;
      return WRITE_OK;
    }
    return pad(new_char, length);
  }

  // Takes a char, copies it into the buffer if there is space, else passes it
  // to the overflow mechanism to be handled separately.
  LIBC_INLINE int write(char new_char) {
    chars_written += 1;
    if (LIBC_LIKELY(wb->buff_cur + 1 <= wb->buff_len)) {
      wb->buff[wb->buff_cur] = new_char;
      wb->buff_cur += 1;
      return WRITE_OK;
    }
    cpp::string_view char_string_view(&new_char, 1);
    return wb->overflow_write(char_string_view);
  }

  LIBC_INLINE int get_chars_written() { return chars_written; }
};

} // namespace printf_core
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STDIO_PRINTF_CORE_WRITER_H
