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

#define FORTRAN_PROCEDURE_NAME(name) name##_

#include <cstddef>
#include <cstdint>

extern "C" {

// CALL FLUSH(n) antedates the Fortran 2003 FLUSH statement.
void FORTRAN_PROCEDURE_NAME(flush)(const int &unit);

// GNU Fortran 77 compatibility function IARGC.
std::int32_t FORTRAN_PROCEDURE_NAME(iargc)();

// GNU Fortran 77 compatibility subroutine GETARG(N, ARG).
void FORTRAN_PROCEDURE_NAME(getarg)(
    std::int32_t &n, std::int8_t *arg, std::int64_t length);

// GNU extension subroutine GETLOG(C).
void FORTRAN_PROCEDURE_NAME(getlog)(std::byte *name, std::int64_t length);

} // extern "C"
#endif // FORTRAN_RUNTIME_EXTENSIONS_H_
