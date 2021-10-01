#if 0 /*===-- include/flang/Runtime/magic-numbers.h -----------------------===*/
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===----------------------------------------------------------------------===*/
#endif
#if 0
This header can be included into both Fortran and C.

This file defines various code values that need to be exported
to predefined Fortran standard modules as well as to C/C++
code in the compiler and runtime library.
These include:
 - the error/end code values that can be returned
   to an IOSTAT= or STAT= specifier on a Fortran I/O statement
   or coindexed data reference (see Fortran 2018 12.11.5,
   16.10.2, and 16.10.2.33)
Codes from <errno.h>, e.g. ENOENT, are assumed to be positive
and are used "raw" as IOSTAT values.

CFI_ERROR_xxx and CFI_INVALID_xxx macros from ISO_Fortran_binding.h
have small positive values.  The FORTRAN_RUNTIME_STAT_xxx macros here
start at 100 so as to never conflict with those codes.
#endif
#ifndef FORTRAN_RUNTIME_MAGIC_NUMBERS_H_
#define FORTRAN_RUNTIME_MAGIC_NUMBERS_H_

#define FORTRAN_RUNTIME_IOSTAT_END (-1)
#define FORTRAN_RUNTIME_IOSTAT_EOR (-2)
#define FORTRAN_RUNTIME_IOSTAT_FLUSH (-3)
#define FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT 256

#define FORTRAN_RUNTIME_STAT_FAILED_IMAGE 101
#define FORTRAN_RUNTIME_STAT_LOCKED 102
#define FORTRAN_RUNTIME_STAT_LOCKED_OTHER_IMAGE 103
#define FORTRAN_RUNTIME_STAT_STOPPED_IMAGE 104
#define FORTRAN_RUNTIME_STAT_UNLOCKED 105
#define FORTRAN_RUNTIME_STAT_UNLOCKED_FAILED_IMAGE 106

#if 0
Status codes for GET_COMMAND_ARGUMENT. The status for 'value too short' needs
to be -1, the others must be positive.
#endif
#define FORTRAN_RUNTIME_STAT_INVALID_ARG_NUMBER 107
#define FORTRAN_RUNTIME_STAT_MISSING_ARG 108
#define FORTRAN_RUNTIME_STAT_VALUE_TOO_SHORT -1
#endif
