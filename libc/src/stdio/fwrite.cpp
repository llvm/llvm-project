//===-- Implementation of fwrite and fwrite_unlocked ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fwrite.h"
#include "src/stdio/FILE.h"
#include "src/threads/mtx_lock.h"
#include "src/threads/mtx_unlock.h"

namespace __llvm_libc {

size_t fwrite_unlocked(const void *__restrict ptr, size_t size, size_t nmeb,
                       __llvm_libc::FILE *__restrict stream) {
  return stream->write(stream, reinterpret_cast<const char *>(ptr),
                       size * nmeb);
}

size_t fwrite(const void *__restrict ptr, size_t size, size_t nmeb,
              __llvm_libc::FILE *__restrict stream) {
  __llvm_libc::mtx_lock(&stream->lock);
  size_t written = fwrite_unlocked(ptr, size, nmeb, stream);
  __llvm_libc::mtx_unlock(&stream->lock);
  return written;
}

} // namespace __llvm_libc
