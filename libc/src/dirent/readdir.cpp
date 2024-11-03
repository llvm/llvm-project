//===-- Implementation of readdir -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "readdir.h"

#include "src/__support/File/dir.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#include <dirent.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(struct ::dirent *, readdir, (::DIR * dir)) {
  auto *d = reinterpret_cast<__llvm_libc::Dir *>(dir);
  auto dirent_val = d->read();
  if (!dirent_val) {
    libc_errno = dirent_val.error();
    return nullptr;
  }
  return dirent_val.value();
}

} // namespace __llvm_libc
