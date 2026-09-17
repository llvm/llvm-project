//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the bare-metal implementation of rename.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/rename.h"

#include "src/__support/OSUtil/io.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, rename, (const char *old_path, const char *new_path)) {
  int error = __llvm_libc_stdio_rename(old_path, new_path);
  if (error < 0) {
    libc_errno = -error;
    return -1;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
