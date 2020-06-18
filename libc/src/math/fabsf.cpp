//===-- Implementation of fabsf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "utils/FPUtil/BasicOperations.h"

namespace __llvm_libc {

float LLVM_LIBC_ENTRYPOINT(fabsf)(float x) { return fputil::abs(x); }

} // namespace __llvm_libc
