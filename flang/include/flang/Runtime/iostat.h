<<<<<<<< HEAD:flang-rt/include/flang-rt/iostat-funcs.h
//===-- include/flang-rt/iostat-funcs.h -------------------------*- C++ -*-===//
|||||||| c4faf0574a3c:flang/include/flang/Runtime/iostat-funcs.h
//===-- include/flang/Runtime/iostat-funcs.h --------------------*- C++ -*-===//
========
//===-- include/flang/Runtime/iostat.h --------------------------*- C++ -*-===//
>>>>>>>> 64bade38fd6a3e5c8c081bae92ce02f58cab799f:flang/include/flang/Runtime/iostat.h
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines the values returned by the runtime for IOSTAT= specifiers
// on I/O statements.

<<<<<<<< HEAD:flang-rt/include/flang-rt/iostat-funcs.h
#ifndef FLANG_RT_IOSTAT_FUNCS_H_
#define FLANG_RT_IOSTAT_FUNCS_H_
|||||||| c4faf0574a3c:flang/include/flang/Runtime/iostat-funcs.h
#ifndef FORTRAN_RUNTIME_IOSTAT_FUNCS_H_
#define FORTRAN_RUNTIME_IOSTAT_FUNCS_H_
========
#ifndef FORTRAN_RUNTIME_IOSTAT_H_
#define FORTRAN_RUNTIME_IOSTAT_H_
>>>>>>>> 64bade38fd6a3e5c8c081bae92ce02f58cab799f:flang/include/flang/Runtime/iostat.h

#include "flang/Common/api-attrs.h"
#include "flang/Runtime/iostat-consts.h"

namespace Fortran::runtime::io {

RT_API_ATTRS const char *IostatErrorString(int);

} // namespace Fortran::runtime::io

#endif /* FORTRAN_RUNTIME_IOSTAT_H_ */
