//===-- include/flang-rt/iostat-funcs.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the values returned by the runtime for IOSTAT= specifiers
// on I/O statements.

#ifndef FLANGRT_IOSTAT_FUNCS_H_
#define FLANGRT_IOSTAT_FUNCS_H_

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/iostat.h"

namespace Fortran::runtime::io {

RT_API_ATTRS const char *IostatErrorString(int);

} // namespace Fortran::runtime::io
#endif /* FLANGRT_IOSTAT_FUNCS_H_ */
