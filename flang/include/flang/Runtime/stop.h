//===-- include/flang/Runtime/stop.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_STOP_H_
#define FORTRAN_RUNTIME_STOP_H_

#include "flang/Runtime/c-or-cpp.h"
#include "flang/Runtime/entry-names.h"
#include "flang/Runtime/extensions.h"
#include <stdlib.h>

FORTRAN_EXTERN_C_BEGIN

// Program-initiated image stop
NORETURN RT_API_ATTRS void RTNAME(StopStatement)(
    int code DEFAULT_VALUE(EXIT_SUCCESS), bool isErrorStop DEFAULT_VALUE(false),
    bool quiet DEFAULT_VALUE(false));
NORETURN RT_API_ATTRS void RTNAME(StopStatementText)(const char *, size_t,
    bool isErrorStop DEFAULT_VALUE(false), bool quiet DEFAULT_VALUE(false));
void RTNAME(PauseStatement)(NO_ARGUMENTS);
void RTNAME(PauseStatementInt)(int);
void RTNAME(PauseStatementText)(const char *, size_t);
NORETURN void RTNAME(FailImageStatement)(NO_ARGUMENTS);
NORETURN void RTNAME(ProgramEndStatement)(NO_ARGUMENTS);

// Extensions
NORETURN void RTNAME(Exit)(int status DEFAULT_VALUE(EXIT_SUCCESS));
RT_OFFLOAD_API_GROUP_BEGIN
NORETURN void RTNAME(Abort)(NO_ARGUMENTS);
RT_OFFLOAD_API_GROUP_END
void FORTRAN_PROCEDURE_NAME(backtrace)(NO_ARGUMENTS);

// Crash with an error message when the program dynamically violates a Fortran
// constraint.
NORETURN RT_API_ATTRS void RTNAME(ReportFatalUserError)(
    const char *message, const char *source, int line);

FORTRAN_EXTERN_C_END

#endif // FORTRAN_RUNTIME_STOP_H_
