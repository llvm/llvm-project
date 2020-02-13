//===-- runtime/iostat.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iostat.h"

namespace Fortran::runtime::io {
const char *IostatErrorString(int iostat) {
  switch (iostat) {
  case IostatOk: return "No error";
  case IostatEnd: return "End of file during input";
  case IostatEor: return "End of record during non-advancing input";
  case IostatUnflushable: return "FLUSH not possible";
  case IostatInquireInternalUnit: return "INQUIRE on internal unit";
  case IostatGenericError:
    return "I/O error";  // dummy value, there's always a message
  case IostatRecordWriteOverrun: return "Excessive output to fixed-size record";
  case IostatRecordReadOverrun: return "Excessive input from fixed-size record";
  case IostatInternalWriteOverrun:
    return "Internal write overran available records";
  case IostatErrorInFormat: return "Invalid FORMAT";
  case IostatErrorInKeyword: return "Bad keyword argument value";
  default: return nullptr;
  }
}

}
