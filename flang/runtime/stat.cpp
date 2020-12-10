//===-- runtime/stat.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stat.h"
#include "descriptor.h"
#include "terminator.h"

namespace Fortran::runtime {
const char *StatErrorString(int stat) {
  switch (stat) {
  case StatOk:
    return "No error";

  case StatBaseNull:
    return "Base address is null";
  case StatBaseNotNull:
    return "Base address is not null";
  case StatInvalidElemLen:
    return "Invalid element length";
  case StatInvalidRank:
    return "Invalid rank";
  case StatInvalidType:
    return "Invalid type";
  case StatInvalidAttribute:
    return "Invalid attribute";
  case StatInvalidExtent:
    return "Invalid extent";
  case StatInvalidDescriptor:
    return "Invalid descriptor";
  case StatMemAllocation:
    return "Memory allocation failed";
  case StatOutOfBounds:
    return "Out of bounds";

  case StatFailedImage:
    return "Failed image";
  case StatLocked:
    return "Locked";
  case StatLockedOtherImage:
    return "Other image locked";
  case StatStoppedImage:
    return "Image stopped";
  case StatUnlocked:
    return "Unlocked";
  case StatUnlockedFailedImage:
    return "Failed image unlocked";

  default:
    return nullptr;
  }
}

int ToErrmsg(Descriptor *errmsg, int stat) {
  if (stat != StatOk && errmsg && errmsg->raw().base_addr &&
      errmsg->type() == TypeCode(TypeCategory::Character, 1) &&
      errmsg->rank() == 0) {
    if (const char *msg{StatErrorString(stat)}) {
      char *buffer{errmsg->OffsetElement()};
      std::size_t bufferLength{errmsg->ElementBytes()};
      std::size_t msgLength{std::strlen(msg)};
      if (msgLength <= bufferLength) {
        std::memcpy(buffer, msg, bufferLength);
      } else {
        std::memcpy(buffer, msg, msgLength);
        std::memset(buffer + msgLength, ' ', bufferLength - msgLength);
      }
    }
  }
  return stat;
}

int ReturnError(
    Terminator &terminator, int stat, Descriptor *errmsg, bool hasStat) {
  if (stat == StatOk || hasStat) {
    return ToErrmsg(errmsg, stat);
  } else if (const char *msg{StatErrorString(stat)}) {
    terminator.Crash(msg);
  } else {
    terminator.Crash("Invalid Fortran runtime STAT= code %d", stat);
  }
  return stat;
}
} // namespace Fortran::runtime
