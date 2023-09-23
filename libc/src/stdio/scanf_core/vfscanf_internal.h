//===-- Internal implementation header of vfscanf ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_SCANF_CORE_VFSCANF_INTERNAL_H
#define LLVM_LIBC_SRC_STDIO_SCANF_CORE_VFSCANF_INTERNAL_H

#include "src/__support/File/file.h"
#include "src/__support/arg_list.h"
#include "src/stdio/scanf_core/reader.h"
#include "src/stdio/scanf_core/scanf_main.h"

#include <stdio.h>

namespace __llvm_libc {

namespace internal {

#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE

LIBC_INLINE int getc(void *f) {
  unsigned char c;
  auto result = reinterpret_cast<__llvm_libc::File *>(f)->read_unlocked(&c, 1);
  size_t r = result.value;
  if (result.has_error() || r != 1)
    return '\0';

  return c;
}

LIBC_INLINE void ungetc(int c, void *f) {
  reinterpret_cast<__llvm_libc::File *>(f)->ungetc(c);
}

LIBC_INLINE int ferror_unlocked(FILE *f) {
  return reinterpret_cast<__llvm_libc::File *>(f)->error_unlocked();
}

#else // defined(LIBC_COPT_STDIO_USE_SYSTEM_FILE)

LIBC_INLINE int getc(void *f) { return ::getc(reinterpret_cast<::FILE *>(f)); }

LIBC_INLINE void ungetc(int c, void *f) {
  ::ungetc(c, reinterpret_cast<::FILE *>(f));
}

LIBC_INLINE int ferror_unlocked(::FILE *f) { return ::ferror_unlocked(f); }

#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE

} // namespace internal

namespace scanf_core {

LIBC_INLINE int vfscanf_internal(::FILE *__restrict stream,
                                 const char *__restrict format,
                                 internal::ArgList &args) {
  scanf_core::Reader reader(stream, &internal::getc, internal::ungetc);
  int retval = scanf_core::scanf_main(&reader, format, args);
  if (retval == 0 && internal::ferror_unlocked(stream))
    return EOF;

  return retval;
}
} // namespace scanf_core
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_SCANF_CORE_VFSCANF_INTERNAL_H
