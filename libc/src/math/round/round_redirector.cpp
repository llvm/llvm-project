//===----------------  Implementation of round redirector -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <math.h>

namespace llvm_libc {

double __round_redirector(double x) {
  return ::round(x);
}

} // namespace llvm_libc
