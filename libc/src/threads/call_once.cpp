//===-- Linux implementation of the call_once function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/call_once.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/callonce.h"

#include <threads.h> // For once_flag and __call_once_func_t definitions.

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void, call_once,
                   (once_flag * flag, __call_once_func_t func)) {
  callonce(reinterpret_cast<CallOnceFlag *>(flag),
           reinterpret_cast<CallOnceCallback *>(func));
}

} // namespace LIBC_NAMESPACE_DECL
