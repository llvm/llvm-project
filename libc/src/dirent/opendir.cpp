//===-- Implementation of opendir -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "opendir.h"

#include "src/__support/File/dir.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"

#include <dirent.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(::DIR *, opendir, (const char *name)) {
  auto dir = Dir::open(name);
  if (!dir) {
    libc_errno = dir.error();
    return nullptr;
  }
  return reinterpret_cast<DIR *>(dir.value());
}

} // namespace LIBC_NAMESPACE_DECL
