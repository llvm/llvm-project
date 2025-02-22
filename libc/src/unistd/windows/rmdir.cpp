//===-- Windows implementation of rmdir -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/rmdir.h"
#include "src/__support/OSUtil/windows/error.h"
#include "src/__support/common.h"
#include "src/errno/libc_errno.h"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, rmdir, (const char *path)) {
  if (::RemoveDirectoryA(path))
    return 0;

  DWORD error = ::GetLastError();
  libc_errno = map_win_error_to_errno(error);
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
