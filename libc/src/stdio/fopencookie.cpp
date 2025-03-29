//===-- Implementation of fopencookie -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopencookie.h"
#include "hdr/stdio_macros.h"
#include "hdr/types/FILE.h"
#include "hdr/types/cookie_io_functions_t.h"
#include "hdr/types/off_t.h"
#include "src/__support/CPP/new.h"
#include "src/__support/File/file.h"

#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

namespace {

class CookieFile : public LIBC_NAMESPACE::File {
  void *cookie;
  cookie_io_functions_t ops;

  static FileIOResult cookie_write(File *f, const void *data, size_t size);
  static FileIOResult cookie_read(File *f, void *data, size_t size);
  static ErrorOr<off_t> cookie_seek(File *f, off_t offset, int whence);
  static int cookie_close(File *f);

public:
  CookieFile(void *c, cookie_io_functions_t cops, uint8_t *buffer,
             size_t bufsize, File::ModeFlags mode)
      : File(&cookie_write, &cookie_read, &CookieFile::cookie_seek,
             &cookie_close, buffer, bufsize, 0 /* default buffering mode */,
             true /* File owns buffer */, mode),
        cookie(c), ops(cops) {}
};

FileIOResult CookieFile::cookie_write(File *f, const void *data, size_t size) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.write == nullptr)
    return 0;
  return static_cast<size_t>(cookie_file->ops.write(
      cookie_file->cookie, reinterpret_cast<const char *>(data), size));
}

FileIOResult CookieFile::cookie_read(File *f, void *data, size_t size) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.read == nullptr)
    return 0;
  return static_cast<size_t>(cookie_file->ops.read(
      cookie_file->cookie, reinterpret_cast<char *>(data), size));
}

ErrorOr<off_t> CookieFile::cookie_seek(File *f, off_t offset, int whence) {
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

int CookieFile::cookie_close(File *f) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.close == nullptr)
    return 0;
  int retval = cookie_file->ops.close(cookie_file->cookie);
  if (retval != 0)
    return retval;
  delete cookie_file;
  return 0;
}

} // anonymous namespace

LLVM_LIBC_FUNCTION(::FILE *, fopencookie,
                   (void *cookie, const char *mode,
                    cookie_io_functions_t ops)) {
  uint8_t *buffer;
  {
    AllocChecker ac;
    buffer = new (ac) uint8_t[File::DEFAULT_BUFFER_SIZE];
    if (!ac)
      return nullptr;
  }
  AllocChecker ac;
  auto *file = new (ac) CookieFile(
      cookie, ops, buffer, File::DEFAULT_BUFFER_SIZE, File::mode_flags(mode));
  if (!ac)
    return nullptr;
  return reinterpret_cast<::FILE *>(file);
}

} // namespace LIBC_NAMESPACE_DECL
