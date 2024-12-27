//===-- Reader definition for scanf -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/scanf_core/reader.h"
#include "hdr/types/FILE.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/config.h"

#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
namespace scanf_core {

namespace internal {

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

} // namespace internal

char Reader::getc() {
  ++cur_chars_read;
  if (rb != nullptr) {
    char output = rb->buffer[rb->buff_cur];
    ++(rb->buff_cur);
    return output;
  }
  // This should reset the buffer if applicable.
  return static_cast<char>(internal::getc(input_stream));
}

void Reader::ungetc(char c) {
  --cur_chars_read;
  if (rb != nullptr && rb->buff_cur > 0) {
    // While technically c should be written back to the buffer, in scanf we
    // always write the character that was already there. Additionally, the
    // buffer is most likely to contain a string that isn't part of a file,
    // which may not be writable.
    --(rb->buff_cur);
    return;
  }
  internal::ungetc(static_cast<int>(c), input_stream);
}

} // namespace scanf_core
} // namespace LIBC_NAMESPACE_DECL
