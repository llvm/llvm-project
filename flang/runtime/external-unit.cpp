//===-- runtime/external-unit.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implemenation of ExternalFileUnit for RT_USE_PSEUDO_FILE_UNIT=0.
//
//===----------------------------------------------------------------------===//

#include "io-error.h"
#include "lock.h"
#include "tools.h"
#include "unit-map.h"
#include "unit.h"

// NOTE: the header files above may define OpenMP declare target
// variables, so they have to be included unconditionally
// so that the offload entries are consistent between host and device.
#if !defined(RT_USE_PSEUDO_FILE_UNIT)

#include <cstdio>
#include <limits>

namespace Fortran::runtime::io {

// The per-unit data structures are created on demand so that Fortran I/O
// should work without a Fortran main program.
static Lock unitMapLock;
static Lock createOpenLock;
static UnitMap *unitMap{nullptr};

void FlushOutputOnCrash(const Terminator &terminator) {
  if (!defaultOutput && !errorOutput) {
    return;
  }
  IoErrorHandler handler{terminator};
  handler.HasIoStat(); // prevent nested crash if flush has error
  CriticalSection critical{unitMapLock};
  if (defaultOutput) {
    defaultOutput->FlushOutput(handler);
  }
  if (errorOutput) {
    errorOutput->FlushOutput(handler);
  }
}

ExternalFileUnit *ExternalFileUnit::LookUp(int unit) {
  return GetUnitMap().LookUp(unit);
}

ExternalFileUnit *ExternalFileUnit::LookUpOrCreate(
    int unit, const Terminator &terminator, bool &wasExtant) {
  return GetUnitMap().LookUpOrCreate(unit, terminator, wasExtant);
}

ExternalFileUnit *ExternalFileUnit::LookUpOrCreateAnonymous(int unit,
    Direction dir, Fortran::common::optional<bool> isUnformatted,
    const Terminator &terminator) {
  // Make sure that the returned anonymous unit has been opened
  // not just created in the unitMap.
  CriticalSection critical{createOpenLock};
  bool exists{false};
  ExternalFileUnit *result{
      GetUnitMap().LookUpOrCreate(unit, terminator, exists)};
  if (result && !exists) {
    IoErrorHandler handler{terminator};
    result->OpenAnonymousUnit(
        dir == Direction::Input ? OpenStatus::Unknown : OpenStatus::Replace,
        Action::ReadWrite, Position::Rewind, Convert::Unknown, handler);
    result->isUnformatted = isUnformatted;
  }
  return result;
}

ExternalFileUnit *ExternalFileUnit::LookUp(
    const char *path, std::size_t pathLen) {
  return GetUnitMap().LookUp(path, pathLen);
}

ExternalFileUnit &ExternalFileUnit::CreateNew(
    int unit, const Terminator &terminator) {
  bool wasExtant{false};
  ExternalFileUnit *result{
      GetUnitMap().LookUpOrCreate(unit, terminator, wasExtant)};
  RUNTIME_CHECK(terminator, result && !wasExtant);
  return *result;
}

ExternalFileUnit *ExternalFileUnit::LookUpForClose(int unit) {
  return GetUnitMap().LookUpForClose(unit);
}

ExternalFileUnit &ExternalFileUnit::NewUnit(
    const Terminator &terminator, bool forChildIo) {
  ExternalFileUnit &unit{GetUnitMap().NewUnit(terminator)};
  unit.createdForInternalChildIo_ = forChildIo;
  return unit;
}

bool ExternalFileUnit::OpenUnit(Fortran::common::optional<OpenStatus> status,
    Fortran::common::optional<Action> action, Position position,
    OwningPtr<char> &&newPath, std::size_t newPathLength, Convert convert,
    IoErrorHandler &handler) {
  if (convert == Convert::Unknown) {
    convert = executionEnvironment.conversion;
  }
  swapEndianness_ = convert == Convert::Swap ||
      (convert == Convert::LittleEndian && !isHostLittleEndian) ||
      (convert == Convert::BigEndian && isHostLittleEndian);
  bool impliedClose{false};
  if (IsConnected()) {
    bool isSamePath{newPath.get() && path() && pathLength() == newPathLength &&
        std::memcmp(path(), newPath.get(), newPathLength) == 0};
    if (status && *status != OpenStatus::Old && isSamePath) {
      handler.SignalError("OPEN statement for connected unit may not have "
                          "explicit STATUS= other than 'OLD'");
      return impliedClose;
    }
    if (!newPath.get() || isSamePath) {
      // OPEN of existing unit, STATUS='OLD' or unspecified, not new FILE=
      newPath.reset();
      return impliedClose;
    }
    // Otherwise, OPEN on open unit with new FILE= implies CLOSE
    DoImpliedEndfile(handler);
    FlushOutput(handler);
    TruncateFrame(0, handler);
    Close(CloseStatus::Keep, handler);
    impliedClose = true;
  }
  if (newPath.get() && newPathLength > 0) {
    if (const auto *already{
            GetUnitMap().LookUp(newPath.get(), newPathLength)}) {
      handler.SignalError(IostatOpenAlreadyConnected,
          "OPEN(UNIT=%d,FILE='%.*s'): file is already connected to unit %d",
          unitNumber_, static_cast<int>(newPathLength), newPath.get(),
          already->unitNumber_);
      return impliedClose;
    }
  }
  set_path(std::move(newPath), newPathLength);
  Open(status.value_or(OpenStatus::Unknown), action, position, handler);
  auto totalBytes{knownSize()};
  if (access == Access::Direct) {
    if (!openRecl) {
      handler.SignalError(IostatOpenBadRecl,
          "OPEN(UNIT=%d,ACCESS='DIRECT'): record length is not known",
          unitNumber());
    } else if (*openRecl <= 0) {
      handler.SignalError(IostatOpenBadRecl,
          "OPEN(UNIT=%d,ACCESS='DIRECT',RECL=%jd): record length is invalid",
          unitNumber(), static_cast<std::intmax_t>(*openRecl));
    } else if (totalBytes && (*totalBytes % *openRecl != 0)) {
      handler.SignalError(IostatOpenBadRecl,
          "OPEN(UNIT=%d,ACCESS='DIRECT',RECL=%jd): record length is not an "
          "even divisor of the file size %jd",
          unitNumber(), static_cast<std::intmax_t>(*openRecl),
          static_cast<std::intmax_t>(*totalBytes));
    }
    recordLength = openRecl;
  }
  endfileRecordNumber.reset();
  currentRecordNumber = 1;
  if (totalBytes && access == Access::Direct && openRecl.value_or(0) > 0) {
    endfileRecordNumber = 1 + (*totalBytes / *openRecl);
  }
  if (position == Position::Append) {
    if (totalBytes) {
      frameOffsetInFile_ = *totalBytes;
    }
    if (access != Access::Stream) {
      if (!endfileRecordNumber) {
        // Fake it so that we can backspace relative from the end
        endfileRecordNumber = std::numeric_limits<std::int64_t>::max() - 2;
      }
      currentRecordNumber = *endfileRecordNumber;
    }
  }
  return impliedClose;
}

void ExternalFileUnit::OpenAnonymousUnit(
    Fortran::common::optional<OpenStatus> status,
    Fortran::common::optional<Action> action, Position position,
    Convert convert, IoErrorHandler &handler) {
  // I/O to an unconnected unit reads/creates a local file, e.g. fort.7
  std::size_t pathMaxLen{32};
  auto path{SizedNew<char>{handler}(pathMaxLen)};
  std::snprintf(path.get(), pathMaxLen, "fort.%d", unitNumber_);
  OpenUnit(status, action, position, std::move(path), std::strlen(path.get()),
      convert, handler);
}

void ExternalFileUnit::CloseUnit(CloseStatus status, IoErrorHandler &handler) {
  DoImpliedEndfile(handler);
  FlushOutput(handler);
  Close(status, handler);
}

void ExternalFileUnit::DestroyClosed() {
  GetUnitMap().DestroyClosed(*this); // destroys *this
}

Iostat ExternalFileUnit::SetDirection(Direction direction) {
  if (direction == Direction::Input) {
    if (mayRead()) {
      direction_ = Direction::Input;
      return IostatOk;
    } else {
      return IostatReadFromWriteOnly;
    }
  } else {
    if (mayWrite()) {
      direction_ = Direction::Output;
      return IostatOk;
    } else {
      return IostatWriteToReadOnly;
    }
  }
}

UnitMap &ExternalFileUnit::CreateUnitMap() {
  Terminator terminator{__FILE__, __LINE__};
  IoErrorHandler handler{terminator};
  UnitMap &newUnitMap{*New<UnitMap>{terminator}().release()};

  bool wasExtant{false};
  ExternalFileUnit &out{*newUnitMap.LookUpOrCreate(
      FORTRAN_DEFAULT_OUTPUT_UNIT, terminator, wasExtant)};
  RUNTIME_CHECK(terminator, !wasExtant);
  out.Predefine(1);
  handler.SignalError(out.SetDirection(Direction::Output));
  out.isUnformatted = false;
  defaultOutput = &out;

  ExternalFileUnit &in{*newUnitMap.LookUpOrCreate(
      FORTRAN_DEFAULT_INPUT_UNIT, terminator, wasExtant)};
  RUNTIME_CHECK(terminator, !wasExtant);
  in.Predefine(0);
  handler.SignalError(in.SetDirection(Direction::Input));
  in.isUnformatted = false;
  defaultInput = &in;

  ExternalFileUnit &error{
      *newUnitMap.LookUpOrCreate(FORTRAN_ERROR_UNIT, terminator, wasExtant)};
  RUNTIME_CHECK(terminator, !wasExtant);
  error.Predefine(2);
  handler.SignalError(error.SetDirection(Direction::Output));
  error.isUnformatted = false;
  errorOutput = &error;

  return newUnitMap;
}

// A back-up atexit() handler for programs that don't terminate with a main
// program END or a STOP statement or other Fortran-initiated program shutdown,
// such as programs with a C main() that terminate normally.  It flushes all
// external I/O units.  It is registered once the first time that any external
// I/O is attempted.
static void CloseAllExternalUnits() {
  IoErrorHandler handler{"Fortran program termination"};
  ExternalFileUnit::CloseAll(handler);
}

UnitMap &ExternalFileUnit::GetUnitMap() {
  if (unitMap) {
    return *unitMap;
  }
  {
    CriticalSection critical{unitMapLock};
    if (unitMap) {
      return *unitMap;
    }
    unitMap = &CreateUnitMap();
  }
  std::atexit(CloseAllExternalUnits);
  return *unitMap;
}

void ExternalFileUnit::CloseAll(IoErrorHandler &handler) {
  CriticalSection critical{unitMapLock};
  if (unitMap) {
    unitMap->CloseAll(handler);
    FreeMemoryAndNullify(unitMap);
  }
  defaultOutput = nullptr;
  defaultInput = nullptr;
  errorOutput = nullptr;
}

void ExternalFileUnit::FlushAll(IoErrorHandler &handler) {
  CriticalSection critical{unitMapLock};
  if (unitMap) {
    unitMap->FlushAll(handler);
  }
}

int ExternalFileUnit::GetAsynchronousId(IoErrorHandler &handler) {
  if (!mayAsynchronous()) {
    handler.SignalError(IostatBadAsynchronous);
    return -1;
  } else {
    for (int j{0}; 64 * j < maxAsyncIds; ++j) {
      if (auto least{asyncIdAvailable_[j].LeastElement()}) {
        asyncIdAvailable_[j].reset(*least);
        return 64 * j + static_cast<int>(*least);
      }
    }
    handler.SignalError(IostatTooManyAsyncOps);
    return -1;
  }
}

bool ExternalFileUnit::Wait(int id) {
  if (static_cast<std::size_t>(id) >= maxAsyncIds ||
      asyncIdAvailable_[id / 64].test(id % 64)) {
    return false;
  } else {
    if (id == 0) { // means "all IDs"
      for (int j{0}; 64 * j < maxAsyncIds; ++j) {
        asyncIdAvailable_[j].set();
      }
      asyncIdAvailable_[0].reset(0);
    } else {
      asyncIdAvailable_[id / 64].set(id % 64);
    }
    return true;
  }
}

} // namespace Fortran::runtime::io

#endif // !defined(RT_USE_PSEUDO_FILE_UNIT)
