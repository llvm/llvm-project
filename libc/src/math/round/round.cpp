//===---------------------- Implementation of round -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/round/round.h"

#include "src/__support/common.h"

namespace llvm_libc {

double __round_redirector(double x);

double LLVM_LIBC_ENTRYPOINT(round)(double x) {
  return __round_redirector(x);
}

} // namespace llvm_libc
