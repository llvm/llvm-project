//===-- include/flang/Runtime/support.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines APIs for runtime support code for lowering.
#ifndef FORTRAN_RUNTIME_SUPPORT_H_
#define FORTRAN_RUNTIME_SUPPORT_H_

#include "flang/Runtime/entry-names.h"
#include <cstddef>
#include <cstdint>

namespace Fortran::runtime {

class Descriptor;

extern "C" {

// Predicate: is the storage described by a Descriptor contiguous in memory?
bool RTDECL(IsContiguous)(const Descriptor &);

} // extern "C"
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_SUPPORT_H_
