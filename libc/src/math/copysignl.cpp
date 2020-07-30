//===-- Implementation of copysignl function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "utils/FPUtil/ManipulationFunctions.h"

namespace __llvm_libc {

long double LLVM_LIBC_ENTRYPOINT(copysignl)(long double x, long double y) {
  return fputil::copysign(x, y);
}

} // namespace __llvm_libc
