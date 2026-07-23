
//===-- Implementation of dlinfo ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlinfo.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: https://github.com/llvm/llvm-project/issues/149911
LLVM_LIBC_FUNCTION(int, dlinfo,
                   (void *__restrict handle, int request,
                    void *__restrict info)) {
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
