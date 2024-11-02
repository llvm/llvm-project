//===-- Implementation of fdopen --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fdopen.h"

#include "src/__support/File/linux/file.h"
#include "src/errno/libc_errno.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(::FILE *, fdopen, (int fd, const char *mode)) {
  auto result = LIBC_NAMESPACE::create_file_from_fd(fd, mode);
  if (!result.has_value()) {
    libc_errno = result.error();
    return nullptr;
  }
  return reinterpret_cast<::FILE *>(result.value());
}

} // namespace LIBC_NAMESPACE
