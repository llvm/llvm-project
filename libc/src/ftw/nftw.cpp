//===-- Implementation of nftw function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ftw/nftw.h"
#include "src/ftw/ftw_impl.h"

#include "src/__support/common.h"
#include "src/__support/libc_errno.h"

#include "hdr/ftw_macros.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, nftw,
                   (const char *DirPath, __nftw_func_t Fn, int FdLimit,
                    int Flags)) {
  ftw_impl::CallbackWrapper Wrapper;
  Wrapper.IsNftw = true;
  Wrapper.NftwFnVal = Fn;
  auto Result =
      ftw_impl::doMergedFtw(DirPath, Wrapper, FdLimit, Flags, 0, 0, nullptr);
  if (!Result) {
    libc_errno = Result.error();
    return -1;
  }
  if (Result.value() == -1 && libc_errno == 0)
    libc_errno = EACCES;
  return Result.value();
}

} // namespace LIBC_NAMESPACE_DECL
