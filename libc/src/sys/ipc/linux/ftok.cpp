//===-- Linux implementation of ftok -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/ipc/ftok.h"

#include "src/__support/common.h"
#include "src/sys/stat/stat.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(key_t, ftok, (const char *path, int id)) {

  // ftok implements based on stat
  struct stat st;
  if (LIBC_NAMESPACE::stat(path, &st) < 0)
    return -1;

  return static_cast<key_t>(((id & 0xff) << 24) |
                            ((static_cast<int>(st.st_dev) & 0xff) << 16) |
                            (static_cast<int>(st.st_ino) & 0xffff));
}

} // namespace LIBC_NAMESPACE_DECL
