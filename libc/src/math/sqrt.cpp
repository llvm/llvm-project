//===-- Implementation of sqrt function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/FPUtil/Sqrt.h"
#include "src/__support/common.h"

namespace __llvm_libc {

double LLVM_LIBC_ENTRYPOINT(sqrt)(double x) { return fputil::sqrt(x); }

} // namespace __llvm_libc
