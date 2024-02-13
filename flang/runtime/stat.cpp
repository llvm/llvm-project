//===-- runtime/stat.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stat.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {
RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS const char *StatErrorString(int stat) {
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

  case StatInvalidArgumentNumber:
    return "Invalid argument number";
  case StatMissingArgument:
    return "Missing argument";
  case StatValueTooShort:
    return "Value too short";

  case StatMissingEnvVariable:
    return "Missing environment variable";

  case StatMoveAllocSameAllocatable:
    return "MOVE_ALLOC passed the same address as to and from";

  case StatBadPointerDeallocation:
    return "DEALLOCATE of a pointer that is not the whole content of a pointer "
           "ALLOCATE";

  default:
    return nullptr;
  }
}

RT_API_ATTRS int ToErrmsg(const Descriptor *errmsg, int stat) {
  if (stat != StatOk && errmsg && errmsg->raw().base_addr &&
      errmsg->type() == TypeCode(TypeCategory::Character, 1) &&
      errmsg->rank() == 0) {
    if (const char *msg{StatErrorString(stat)}) {
      char *buffer{errmsg->OffsetElement()};
      std::size_t bufferLength{errmsg->ElementBytes()};
      std::size_t msgLength{Fortran::runtime::strlen(msg)};
      if (msgLength >= bufferLength) {
        std::memcpy(buffer, msg, bufferLength);
      } else {
        std::memcpy(buffer, msg, msgLength);
        std::memset(buffer + msgLength, ' ', bufferLength - msgLength);
      }
    }
  }
  return stat;
}

RT_API_ATTRS int ReturnError(
    Terminator &terminator, int stat, const Descriptor *errmsg, bool hasStat) {
  if (stat == StatOk || hasStat) {
    return ToErrmsg(errmsg, stat);
  } else if (const char *msg{StatErrorString(stat)}) {
    terminator.Crash(msg);
  } else {
    terminator.Crash("Invalid Fortran runtime STAT= code %d", stat);
  }
  return stat;
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
