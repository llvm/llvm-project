//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Linux implementation of execvp
///
//===----------------------------------------------------------------------===//

#include "src/unistd/execvp.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/unistd/environ.h"
#include "src/unistd/linux/exec_utils.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, execvp, (const char *file, char *const argv[])) {
  LIBC_CRASH_ON_NULLPTR(file);
  LIBC_CRASH_ON_NULLPTR(argv);

  auto ret = internal::execvpe(
      file, argv, const_cast<char *const *>(LIBC_NAMESPACE::environ));
  if (!ret) {
    libc_errno = ret.error();
    return -1;
  }
  return *ret;
}

} // namespace LIBC_NAMESPACE_DECL
