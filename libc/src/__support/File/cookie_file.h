//===--- A platform independent cookie file class -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/cookie_io_functions_t.h"
#include "hdr/types/off_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/File/file.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

class CookieFile : public LIBC_NAMESPACE::File {
  void *cookie;
  cookie_io_functions_t ops;

  LIBC_INLINE static FileIOResult cookie_write(File *f, const void *data,
                                               size_t size);
  LIBC_INLINE static FileIOResult cookie_read(File *f, void *data, size_t size);
  LIBC_INLINE static ErrorOr<off_t> cookie_seek(File *f, off_t offset,
                                                int whence);
  LIBC_INLINE static int cookie_close(File *f);

public:
  LIBC_INLINE CookieFile(void *c, cookie_io_functions_t cops, uint8_t *buffer,
                         size_t bufsize, int buffer_mode, File::ModeFlags mode)
      : File(&cookie_write, &cookie_read, &CookieFile::cookie_seek,
             &cookie_close, buffer, bufsize, buffer_mode,
             true /* File owns buffer */, mode),
        cookie(c), ops(cops) {}
};

LIBC_INLINE FileIOResult CookieFile::cookie_write(File *f, const void *data,
                                                  size_t size) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.write == nullptr)
    return 0;
  return static_cast<size_t>(cookie_file->ops.write(
      cookie_file->cookie, reinterpret_cast<const char *>(data), size));
}

LIBC_INLINE FileIOResult CookieFile::cookie_read(File *f, void *data,
                                                 size_t size) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.read == nullptr)
    return 0;
  return static_cast<size_t>(cookie_file->ops.read(
      cookie_file->cookie, reinterpret_cast<char *>(data), size));
}

LIBC_INLINE ErrorOr<off_t> CookieFile::cookie_seek(File *f, off_t offset,
                                                   int whence) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.seek == nullptr) {
    return Error(EINVAL);
  }
  off64_t offset64 = offset;
  int result = cookie_file->ops.seek(cookie_file->cookie, &offset64, whence);
  if (result == 0)
    return offset64;
  return -1;
}

LIBC_INLINE int CookieFile::cookie_close(File *f) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.close == nullptr)
    return 0;
  int retval = cookie_file->ops.close(cookie_file->cookie);
  if (retval != 0)
    return retval;
  delete cookie_file;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
