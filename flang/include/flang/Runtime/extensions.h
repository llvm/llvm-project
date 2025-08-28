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

#include "flang/Runtime/entry-names.h"
#include <cstddef>
#include <cstdint>

#define FORTRAN_PROCEDURE_NAME(name) name##_

#ifdef _WIN32
// UID and GID don't exist on Windows, these exist to avoid errors.
typedef std::uint32_t uid_t;
typedef std::uint32_t gid_t;
#else
#include "sys/types.h" //pid_t
#endif

extern "C" {

// CALL FLUSH(n) antedates the Fortran 2003 FLUSH statement.
void FORTRAN_PROCEDURE_NAME(flush)(const int &unit);

// GNU extension subroutine FDATE
void FORTRAN_PROCEDURE_NAME(fdate)(char *string, std::int64_t length);

void RTNAME(Free)(std::intptr_t ptr);

// Common extensions FSEEK & FTELL, variously named
std::int32_t RTNAME(Fseek)(int unit, std::int64_t zeroBasedPos, int whence,
    const char *sourceFileName, int lineNumber);
std::int64_t RTNAME(Ftell)(int unit);

// GNU Fortran 77 compatibility function IARGC.
std::int32_t FORTRAN_PROCEDURE_NAME(iargc)();

// GNU Fortran 77 compatibility subroutine GETARG(N, ARG).
void FORTRAN_PROCEDURE_NAME(getarg)(
    std::int32_t &n, char *arg, std::int64_t length);

// Calls getgid()
gid_t RTNAME(GetGID)();

// Calls getuid()
uid_t RTNAME(GetUID)();

// GNU extension subroutine GETLOG(C).
void FORTRAN_PROCEDURE_NAME(getlog)(char *name, std::int64_t length);

// GNU extension subroutine HOSTNM(C)
int FORTRAN_PROCEDURE_NAME(hostnm)(char *hn, int length);

std::intptr_t RTNAME(Malloc)(std::size_t size);

// GNU extension function STATUS = SIGNAL(number, handler)
std::int64_t RTNAME(Signal)(std::int64_t number, void (*handler)(int));

// GNU extension subroutine SLEEP(SECONDS)
void RTNAME(Sleep)(std::int64_t seconds);

// GNU extension function TIME()
std::int64_t RTNAME(time)();

// GNU extension function ACCESS(NAME, MODE)
// TODO: not supported on Windows
#ifndef _WIN32
std::int64_t FORTRAN_PROCEDURE_NAME(access)(const char *name,
    std::int64_t nameLength, const char *mode, std::int64_t modeLength);
#endif

// GNU extension subroutine CHDIR(NAME, [STATUS])
int RTNAME(Chdir)(const char *name);

// GNU extension function IERRNO()
int FORTRAN_PROCEDURE_NAME(ierrno)();

// GNU extension subroutine PERROR(STRING)
void RTNAME(Perror)(const char *str);

// MCLOCK -- returns accumulated time in ticks
int FORTRAN_PROCEDURE_NAME(mclock)();

// GNU extension subroutine SECNDS(refTime)
float FORTRAN_PROCEDURE_NAME(secnds)(float *refTime);
float RTNAME(Secnds)(float *refTime, const char *sourceFile, int line);

} // extern "C"
#endif // FORTRAN_RUNTIME_EXTENSIONS_H_
