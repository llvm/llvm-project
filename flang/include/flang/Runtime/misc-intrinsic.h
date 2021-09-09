//===-- include/flang/Runtime/misc-intrinsic.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Miscellaneous intrinsic procedures

#ifndef FORTRAN_RUNTIME_MISC_INTRINSIC_H_
#define FORTRAN_RUNTIME_MISC_INTRINSIC_H_

#include "flang/Runtime/entry-names.h"
#include <cstdint>

namespace Fortran::runtime {

class Descriptor;

extern "C" {
void RTNAME(Transfer)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line);
void RTNAME(TransferSize)(Descriptor &result, const Descriptor &source,
    const Descriptor &mold, const char *sourceFile, int line,
    std::int64_t size);
} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_MISC_INTRINSIC_H_
