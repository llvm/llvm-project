//===-- Implementation of dladdr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dladdr.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: https:// github.com/llvm/llvm-project/issues/97929
LLVM_LIBC_FUNCTION(int, dladdr,
                   ([[maybe_unused]] const void *__restrict addr,
                    [[maybe_unused]] Dl_info *__restrict info)) {
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
