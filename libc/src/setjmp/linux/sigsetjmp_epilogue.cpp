//===-- Implementation of sigsetjmp_epilogue ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/sigsetjmp_epilogue.h"
#include "src/__support/OSUtil/linux/sigmask.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {
[[gnu::returns_twice]] int sigsetjmp_epilogue(sigjmp_buf buffer, int retval) {
  if (retval)
    signals::set_mask(buffer->sigmask);
  else
    signals::get_mask(buffer->sigmask);
  return retval;
}
} // namespace LIBC_NAMESPACE_DECL
