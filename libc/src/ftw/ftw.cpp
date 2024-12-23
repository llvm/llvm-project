//===-- Implementation of ftw function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ftw/ftw.h"
#include "hdr/ftw_macros.h"
#include "src/ftw/ftw_impl.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ftw,
                   (const char *DirPath, __ftw_func_t Fn, int FdLimit)) {
  ftw_impl::CallbackWrapper Wrapper;
  Wrapper.IsNftw = false;
  Wrapper.FtwFnVal = Fn;
  auto Result =
      ftw_impl::doMergedFtw(DirPath, Wrapper, FdLimit, 0, 0, 0, nullptr);
  if (!Result) {
    libc_errno = Result.error();
    return -1;
  }
  return Result.value();
}

} // namespace LIBC_NAMESPACE_DECL
