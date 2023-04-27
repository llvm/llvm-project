//===------------------- Implementation of _Exit --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/quick_exit.h"
#include "src/__support/common.h"

#include "src/stdlib/_Exit.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, _Exit, (int status)) {
  quick_exit(status);
  __builtin_unreachable();
}

} // namespace __llvm_libc
