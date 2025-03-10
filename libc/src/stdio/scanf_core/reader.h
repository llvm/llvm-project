//===-- Reader definition for scanf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H

#include "hdr/types/FILE.h"

#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
#include "src/__support/File/file.h"
#endif

#if defined(LIBC_TARGET_ARCH_IS_GPU)
#include "src/stdio/getc.h"
#include "src/stdio/ungetc.h"
#endif

#include "src/__support/macros/attributes.h" // For LIBC_INLINE
#include "src/__support/macros/config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace scanf_core {
// We use the name "reader_internal" over "internal" because
// "internal" causes name lookups in files that include the current header to be
// ambigious i.e. `internal::foo` in those files, will try to lookup in
// `LIBC_NAMESPACE::scanf_core::internal` over `LIBC_NAMESPACE::internal` for
// e.g., `internal::ArgList` in `libc/src/stdio/scanf_core/scanf_main.h`
namespace reader_internal {

#if defined(LIBC_TARGET_ARCH_IS_GPU)
// The GPU build provides FILE access through the host operating system's
// library. So here we simply use the public entrypoints like in the SYSTEM_FILE
// interface. Entrypoints should normally not call others, this is an exception.
// FIXME: We do not acquire any locks here, so this is not thread safe.
LIBC_INLINE int getc(void *f) {
  return LIBC_NAMESPACE::getc(reinterpret_cast<::FILE *>(f));
}

LIBC_INLINE void ungetc(int c, void *f) {
  LIBC_NAMESPACE::ungetc(c, reinterpret_cast<::FILE *>(f));
}

#elif !defined(LIBC_COPT_STDIO_USE_SYSTEM_FILE)

LIBC_INLINE int getc(void *f) {
  unsigned char c;
  auto result =
      reinterpret_cast<LIBC_NAMESPACE::File *>(f)->read_unlocked(&c, 1);
  size_t r = result.value;
  if (result.has_error() || r != 1)
    return '\0';

  return c;
}

LIBC_INLINE void ungetc(int c, void *f) {
  reinterpret_cast<LIBC_NAMESPACE::File *>(f)->ungetc_unlocked(c);
}

#else  // defined(LIBC_COPT_STDIO_USE_SYSTEM_FILE)

// Since ungetc_unlocked isn't always available, we don't acquire the lock for
// system files.
LIBC_INLINE int getc(void *f) { return ::getc(reinterpret_cast<::FILE *>(f)); }

LIBC_INLINE void ungetc(int c, void *f) {
  ::ungetc(c, reinterpret_cast<::FILE *>(f));
}
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE

} // namespace reader_internal

// This is intended to be either a raw string or a buffer syncronized with the
// file's internal buffer.
struct ReadBuffer {
  const char *buffer;
  size_t buff_len;
  size_t buff_cur = 0;
};

class Reader {
  ReadBuffer *rb;
  void *input_stream = nullptr;
  size_t cur_chars_read = 0;

public:
  // TODO: Set buff_len with a proper constant
  LIBC_INLINE Reader(ReadBuffer *string_buffer) : rb(string_buffer) {}

  LIBC_INLINE Reader(void *stream, ReadBuffer *stream_buffer = nullptr)
      : rb(stream_buffer), input_stream(stream) {}

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
    return static_cast<char>(reader_internal::getc(input_stream));
  }

  // This moves the input back by one character, placing c into the buffer if
  // this is a file reader, else c is ignored.
  LIBC_INLINE void ungetc(char c) {
    --cur_chars_read;
    if (rb != nullptr && rb->buff_cur > 0) {
      // While technically c should be written back to the buffer, in scanf we
      // always write the character that was already there. Additionally, the
      // buffer is most likely to contain a string that isn't part of a file,
      // which may not be writable.
      --(rb->buff_cur);
      return;
    }
    reader_internal::ungetc(static_cast<int>(c), input_stream);
  }

  LIBC_INLINE size_t chars_read() { return cur_chars_read; }
};

} // namespace scanf_core
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_READER_H
