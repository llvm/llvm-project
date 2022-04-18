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
#include "flang/Runtime/magic-numbers.h"
namespace Fortran::runtime::io {

// The value of IOSTAT= is zero when no error, end-of-record,
// or end-of-file condition has arisen; errors are positive values.
// (See 12.11.5 in Fortran 2018 for the complete requirements;
// these constants must match the values of their corresponding
// named constants in the predefined module ISO_FORTRAN_ENV, so
// they're actually defined in another magic-numbers.h header file
// so that they can be included both here and there.)
enum Iostat {
  IostatOk = 0, // no error, EOF, or EOR condition

  // These error codes are required by Fortran (see 12.10.2.16-17) to be
  // negative integer values
  IostatEnd = FORTRAN_RUNTIME_IOSTAT_END, // end-of-file on input & no error
  // End-of-record on non-advancing input, no EOF or error
  IostatEor = FORTRAN_RUNTIME_IOSTAT_EOR,

  // This value is also required to be negative (12.11.5 bullet 6).
  // It signifies a FLUSH statement on an unflushable unit.
  IostatUnflushable = FORTRAN_RUNTIME_IOSTAT_FLUSH,

  // Other errors are positive.  We use "errno" values unchanged.
  // This error is exported in ISO_Fortran_env.
  IostatInquireInternalUnit = FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT,

  // The remaining error codes are not exported.
  IostatGenericError = 1001, // see IOMSG= for details
  IostatRecordWriteOverrun,
  IostatRecordReadOverrun,
  IostatInternalWriteOverrun,
  IostatErrorInFormat,
  IostatErrorInKeyword,
  IostatEndfileDirect,
  IostatEndfileUnwritable,
  IostatOpenBadRecl,
  IostatOpenUnknownSize,
  IostatOpenBadAppend,
  IostatWriteToReadOnly,
  IostatReadFromWriteOnly,
  IostatBackspaceNonSequential,
  IostatBackspaceAtFirstRecord,
  IostatRewindNonSequential,
  IostatWriteAfterEndfile,
  IostatFormattedIoOnUnformattedUnit,
  IostatUnformattedIoOnFormattedUnit,
  IostatListIoOnDirectAccessUnit,
  IostatUnformattedChildOnFormattedParent,
  IostatFormattedChildOnUnformattedParent,
  IostatChildInputFromOutputParent,
  IostatChildOutputToInputParent,
  IostatShortRead,
  IostatMissingTerminator,
  IostatBadUnformattedRecord,
  IostatUTF8Decoding,
  IostatUnitOverflow,
  IostatBadRealInput,
};

const char *IostatErrorString(int);

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IOSTAT_H_
