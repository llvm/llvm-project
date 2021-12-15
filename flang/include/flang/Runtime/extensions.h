//===-- include/flang/Runtime/extensions.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These C-coded entry points with Fortran-mangled names implement legacy
// extensions that will eventually be implemented in Fortran.

#ifndef FORTRAN_RUNTIME_EXTENSIONS_H_
#define FORTRAN_RUNTIME_EXTENSIONS_H_

#define FORTRAN_SUBROUTINE_NAME(name) name##_

extern "C" {

// CALL FLUSH(n) antedates the Fortran 2003 FLUSH statement.
void FORTRAN_SUBROUTINE_NAME(flush)(const int &unit);

} // extern "C"
#endif // FORTRAN_RUNTIME_EXTENSIONS_H_
