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

#include <dirent.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(::DIR *, opendir, (const char *name)) {
  return reinterpret_cast<::DIR *>(Dir::open(name));
}

} // namespace __llvm_libc
