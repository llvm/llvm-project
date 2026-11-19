//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Line-oriented reader for File streams.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_LINE_READER_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_LINE_READER_H

#include "hdr/errno_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/span.h"
#include "src/__support/File/file.h"
#include "src/__support/dynamic_buffer.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

// Struct to hold the result of a line read operation.
struct ReadLineResult {
  // Byte length of the returned line after stripping any trailing '\n' and
  // excluding the terminating '\0'. Embedded '\0' bytes and preceding '\r'
  // bytes are preserved and included in this count.
  size_t bytes_read;
  bool truncated;
  // True only when zero raw bytes were read because the stream was already at
  // EOF. A blank line ("\n") or a final line without a trailing newline
  // consumes at least one raw byte and returns eof == false (with
  // bytes_read == 0 for "\n"); the following call returns eof == true.
  bool eof;
};

// Reads a single line from the given file into the provided buffer, stripping
// any trailing '\n' and ensuring the result is null-terminated. A line too
// long for the buffer is reported as truncated after draining the remainder of
// the line up to '\n' or EOF.
//
// Note: POSIX getline/getdelim cannot be used here because bounded callers
// must operate in-place within a fixed buffer without dynamic heap
// allocations. See the DynamicBuffer overload for callers that own their
// buffer and may grow it.
[[nodiscard]] LIBC_INLINE ErrorOr<ReadLineResult>
read_line(File *f, cpp::span<char> buf) {
  if (!f)
    return Error(EINVAL);
  if (buf.size() < 2)
    return Error(ERANGE);

  File::FileLock lock(f);
  size_t bytes_read = 0;
  bool found_newline = false;
  bool saw_eof = false;
  bool truncated = false;

  for (char &ch : buf.first(buf.size() - 1)) {
    FileIOResult result = f->read_unlocked(&ch, 1);
    if (result.has_error())
      return Error(result.error);
    if (result.value != 1) {
      saw_eof = true;
      break;
    }
    ++bytes_read;
    if (ch == '\n') {
      found_newline = true;
      break;
    }
  }

  if (!found_newline && !saw_eof) {
    char c = '\0';
    while (true) {
      FileIOResult result = f->read_unlocked(&c, 1);
      if (result.has_error())
        return Error(result.error);
      if (result.value != 1 || c == '\n')
        break;
      truncated = true;
    }
  }

  if (f->error_unlocked())
    return Error(EIO);

  bool eof = (bytes_read == 0);

  // If the line ended with a newline, strip it.
  if (found_newline)
    --bytes_read;

  buf[bytes_read] = '\0';
  return ReadLineResult{bytes_read, truncated, eof};
}

// Reads a single line into a caller-owned buffer, growing it as needed so
// that arbitrarily long lines can be read. Otherwise behaves as the
// fixed-buffer read_line overload; the result is never truncated. Does not
// allocate when the stream is immediately at EOF (so an empty DynamicBuffer
// stays unallocated when eof is true). On allocation failure, drains the rest
// of the current line up to '\n' or EOF before returning ENOMEM.
[[nodiscard]] LIBC_INLINE ErrorOr<ReadLineResult>
read_line(File *f, DynamicBuffer &buf) {
  if (!f)
    return Error(EINVAL);

  File::FileLock lock(f);
  size_t bytes_read = 0;
  bool found_newline = false;

  while (true) {
    char ch = '\0';
    FileIOResult result = f->read_unlocked(&ch, 1);
    if (result.has_error())
      return Error(result.error);
    if (result.value != 1)
      break;

    // One byte for the character just read, one for the terminator.
    if (LIBC_UNLIKELY(bytes_read > cpp::numeric_limits<size_t>::max() - 2) ||
        !buf.reserve(bytes_read + 2)) {
      if (ch != '\n') {
        char c = '\0';
        while (true) {
          FileIOResult drain = f->read_unlocked(&c, 1);
          if (drain.has_error())
            return Error(drain.error);
          if (drain.value != 1 || c == '\n')
            break;
        }
      }
      return Error(ENOMEM);
    }

    buf.span()[bytes_read++] = ch;
    if (ch == '\n') {
      found_newline = true;
      break;
    }
  }

  if (f->error_unlocked())
    return Error(EIO);

  bool eof = (bytes_read == 0);

  // If the line ended with a newline, strip it.
  if (found_newline)
    --bytes_read;

  if (!buf.span().empty())
    buf.span()[bytes_read] = '\0';
  return ReadLineResult{bytes_read, /*truncated=*/false, eof};
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_LINE_READER_H
