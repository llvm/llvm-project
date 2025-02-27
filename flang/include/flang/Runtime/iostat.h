//===-- include/flang/Runtime/iostat.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the values returned by the runtime for IOSTAT= specifiers
// on I/O statements.

#ifndef FORTRAN_RUNTIME_IOSTAT_H_
#define FORTRAN_RUNTIME_IOSTAT_H_

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/iostat-consts.h"

namespace Fortran::runtime::io {

RT_API_ATTRS const char *IostatErrorString(int);

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IOSTAT_H_
