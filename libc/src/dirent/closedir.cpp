//===-- Implementation of closedir ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "closedir.h"

#include "src/__support/File/dir.h"
#include "src/__support/common.h"

#include <dirent.h>
#include <errno.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, closedir, (::DIR * dir)) {
  auto *d = reinterpret_cast<__llvm_libc::Dir *>(dir);
  int retval = d->close();
  if (retval != 0) {
    errno = retval;
    return -1;
  }
  return 0;
}

} // namespace __llvm_libc
