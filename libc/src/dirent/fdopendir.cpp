//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the POSIX fdopendir function.
///
//===----------------------------------------------------------------------===//

#include "fdopendir.h"

#include "src/__support/File/dir.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(DIR *, fdopendir, (int fd)) {
  int check = platform_check_dir(fd);
  if (check != 0) {
    libc_errno = check;
    return nullptr;
  }

  auto dir = Dir::fdopen(fd);
  if (!dir) {
    libc_errno = dir.error();
    return nullptr;
  }
  return reinterpret_cast<DIR *>(dir.value());
}

} // namespace LIBC_NAMESPACE_DECL
