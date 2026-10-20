//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of ftw.
///
//===----------------------------------------------------------------------===//

#include "src/ftw/ftw.h"

#include "hdr/types/__ftw_func_t.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/null_check.h"
#include "src/ftw/ftw_impl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ftw,
                   (const char *dir_path, __ftw_func_t fn, int fd_limit)) {
  LIBC_CRASH_ON_NULLPTR(dir_path);
  LIBC_CRASH_ON_NULLPTR(fn);

  auto result = internal::ftw_impl(dir_path, fn, fd_limit);
  if (!result.has_value()) {
    libc_errno = result.error();
    return -1;
  }
  return result.value();
}

} // namespace LIBC_NAMESPACE_DECL
