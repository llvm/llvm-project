//===-- Implementation of fopencookie -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fopencookie.h"
#include "src/__support/File/file.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

namespace __llvm_libc {

namespace {

class CookieFile : public __llvm_libc::File {
public:
  void *cookie;
  cookie_io_functions_t ops;
};

size_t write_func(File *f, const void *data, size_t size) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.write == nullptr)
    return 0;
  return cookie_file->ops.write(cookie_file->cookie,
                                reinterpret_cast<const char *>(data), size);
}

size_t read_func(File *f, void *data, size_t size) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.read == nullptr)
    return 0;
  return cookie_file->ops.read(cookie_file->cookie,
                               reinterpret_cast<char *>(data), size);
}

int seek_func(File *f, long offset, int whence) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.seek == nullptr) {
    errno = EINVAL;
    return -1;
  }
  off64_t offset64 = offset;
  return cookie_file->ops.seek(cookie_file->cookie, &offset64, whence);
}

int close_func(File *f) {
  auto cookie_file = reinterpret_cast<CookieFile *>(f);
  if (cookie_file->ops.close == nullptr)
    return 0;
  return cookie_file->ops.close(cookie_file->cookie);
}

int flush_func(File *) { return 0; }

} // anonymous namespace

LLVM_LIBC_FUNCTION(::FILE *, fopencookie,
                   (void *cookie, const char *mode,
                    cookie_io_functions_t ops)) {
  auto modeflags = File::mode_flags(mode);
  void *buffer = malloc(File::DEFAULT_BUFFER_SIZE);
  auto *file = reinterpret_cast<CookieFile *>(malloc(sizeof(CookieFile)));
  if (file == nullptr)
    return nullptr;

  File::init(file, &write_func, &read_func, &seek_func, &close_func,
             &flush_func, buffer, File::DEFAULT_BUFFER_SIZE,
             0,    // Default buffering style
             true, // Owned buffer
             modeflags);
  file->cookie = cookie;
  file->ops = ops;

  return reinterpret_cast<::FILE *>(file);
}

} // namespace __llvm_libc
