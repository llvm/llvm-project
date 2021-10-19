//===-- Implementation of the tan function for x86_64 ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tan.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, tan, (double x)) {
  double result;
  double one;
  // The fptan instruction pushes the number 1 on to the FP stack after
  // computing tan. So, we read out the one before popping the actual result.
  __asm__ __volatile__("fptan" : "=t"(one) : "f"(x));
  __asm__ __volatile__("fstpl %0" : "=m"(result));
  return result;
}

} // namespace __llvm_libc
