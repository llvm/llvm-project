#ifndef LLVM_LIBC_SRC_STDIO_INLINE_GETLINE_H
#define LLVM_LIBC_SRC_STDIO_INLINE_GETLINE_H

#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/func/realloc.h"
#include "hdr/types/FILE.h"
#include "hdr/types/size_t.h"
#include "hdr/types/ssize_t.h"
#include "src/__support/File/file.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

constexpr int INIT_BASE = 32;

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE ssize_t __getline(char **__restrict lineptr, size_t *__restrict n,
                               int del, ::FILE *__restrict stream) {
  if (!lineptr || !n || !stream) {
    libc_errno = EINVAL;
    return -1;
  }

  auto *file = reinterpret_cast<LIBC_NAMESPACE::File *>(stream);

  if (*lineptr == nullptr) {
    *n = (*n == 0) ? INIT_BASE : *n;
    *lineptr = static_cast<char *>(malloc(*n));
    if (!*lineptr) {
      libc_errno = ENOMEM;
      return -1;
    }
  }
  uint8_t c = 0;
  size_t bytes_read = 0;
  file->lock();
  while (true) {
    auto result = file->read_unlocked(&c, 1);
    if (result.has_error()) {
      file->unlock();
      libc_errno = result.error;
      return EOF;
    }
    if (result.value != 1)
      break;
    if ((bytes_read + 2) > *n) {
      size_t new_size =
          ((bytes_read + 2) >= (*n * 2)) ? bytes_read + 2 : *n * 2;
      char *tmpptr = static_cast<char *>(realloc(*lineptr, new_size));
      if (!tmpptr) {
        file->unlock();
        libc_errno = ENOMEM;
        return -1;
      }
      *lineptr = tmpptr;
      *n = new_size;
    }
    (*lineptr)[bytes_read++] = static_cast<char>(c);

    if (c == del)
      break;
  }
  file->unlock();
  if (bytes_read == 0)
    return -1;
  (*lineptr)[bytes_read] = '\0';
  return bytes_read;
}
} // namespace LIBC_NAMESPACE_DECL
#endif
