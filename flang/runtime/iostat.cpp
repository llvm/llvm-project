//===-- runtime/iostat.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/iostat.h"

namespace Fortran::runtime::io {
const char *IostatErrorString(int iostat) {
  switch (iostat) {
  case IostatOk:
    return "No error";
  case IostatEnd:
    return "End of file during input";
  case IostatEor:
    return "End of record during non-advancing input";
  case IostatUnflushable:
    return "FLUSH not possible";
  case IostatInquireInternalUnit:
    return "INQUIRE on internal unit";
  case IostatGenericError:
    return "I/O error"; // dummy value, there's always a message
  case IostatRecordWriteOverrun:
    return "Excessive output to fixed-size record";
  case IostatRecordReadOverrun:
    return "Excessive input from fixed-size record";
  case IostatInternalWriteOverrun:
    return "Internal write overran available records";
  case IostatErrorInFormat:
    return "Bad FORMAT";
  case IostatErrorInKeyword:
    return "Bad keyword argument value";
  case IostatEndfileDirect:
    return "ENDFILE on direct-access file";
  case IostatEndfileUnwritable:
    return "ENDFILE on read-only file";
  case IostatOpenBadRecl:
    return "OPEN with bad RECL= value";
  case IostatOpenUnknownSize:
    return "OPEN of file of unknown size";
  case IostatOpenBadAppend:
    return "OPEN(POSITION='APPEND') of unpositionable file";
  case IostatWriteToReadOnly:
    return "Attempted output to read-only file";
  case IostatReadFromWriteOnly:
    return "Attempted input from write-only file";
  case IostatBackspaceNonSequential:
    return "BACKSPACE on non-sequential file";
  case IostatBackspaceAtFirstRecord:
    return "BACKSPACE at first record";
  case IostatRewindNonSequential:
    return "REWIND on non-sequential file";
  case IostatWriteAfterEndfile:
    return "WRITE after ENDFILE";
  case IostatFormattedIoOnUnformattedUnit:
    return "Formatted I/O on unformatted file";
  case IostatUnformattedIoOnFormattedUnit:
    return "Unformatted I/O on formatted file";
  case IostatListIoOnDirectAccessUnit:
    return "List-directed or NAMELIST I/O on direct-access file";
  case IostatUnformattedChildOnFormattedParent:
    return "Unformatted child I/O on formatted parent unit";
  case IostatFormattedChildOnUnformattedParent:
    return "Formatted child I/O on unformatted parent unit";
  case IostatChildInputFromOutputParent:
    return "Child input from output parent unit";
  case IostatChildOutputToInputParent:
    return "Child output to input parent unit";
  case IostatShortRead:
    return "Read from external unit returned insufficient data";
  case IostatMissingTerminator:
    return "Sequential record missing its terminator";
  case IostatBadUnformattedRecord:
    return "Erroneous unformatted sequential file record structure";
  case IostatUTF8Decoding:
    return "UTF-8 decoding error";
  case IostatUnitOverflow:
    return "UNIT number is out of range";
  case IostatBadRealInput:
    return "Bad REAL input value";
  default:
    return nullptr;
  }
}

} // namespace Fortran::runtime::io
