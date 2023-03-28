//===-- runtime/io-api.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the I/O statement API

#include "flang/Runtime/io-api.h"
#include "descriptor-io.h"
#include "edit-input.h"
#include "edit-output.h"
#include "environment.h"
#include "format.h"
#include "io-stmt.h"
#include "terminator.h"
#include "tools.h"
#include "unit.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/memory.h"
#include <cstdlib>
#include <memory>

namespace Fortran::runtime::io {

const char *InquiryKeywordHashDecode(
    char *buffer, std::size_t n, InquiryKeywordHash hash) {
  if (n < 1) {
    return nullptr;
  }
  char *p{buffer + n};
  *--p = '\0';
  while (hash > 1) {
    if (p < buffer) {
      return nullptr;
    }
    *--p = 'A' + (hash % 26);
    hash /= 26;
  }
  return hash == 1 ? p : nullptr;
}

template <Direction DIR>
Cookie BeginInternalArrayListIO(const Descriptor &descriptor,
    void ** /*scratchArea*/, std::size_t /*scratchBytes*/,
    const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalListIoStatementState<DIR>>{oom}(
      descriptor, sourceFile, sourceLine)
              .release()
              ->ioStatementState();
}

Cookie IONAME(BeginInternalArrayListOutput)(const Descriptor &descriptor,
    void **scratchArea, std::size_t scratchBytes, const char *sourceFile,
    int sourceLine) {
  return BeginInternalArrayListIO<Direction::Output>(
      descriptor, scratchArea, scratchBytes, sourceFile, sourceLine);
}

Cookie IONAME(BeginInternalArrayListInput)(const Descriptor &descriptor,
    void **scratchArea, std::size_t scratchBytes, const char *sourceFile,
    int sourceLine) {
  return BeginInternalArrayListIO<Direction::Input>(
      descriptor, scratchArea, scratchBytes, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginInternalArrayFormattedIO(const Descriptor &descriptor,
    const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, void ** /*scratchArea*/,
    std::size_t /*scratchBytes*/, const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalFormattedIoStatementState<DIR>>{oom}(descriptor, format,
      formatLength, formatDescriptor, sourceFile, sourceLine)
              .release()
              ->ioStatementState();
}

Cookie IONAME(BeginInternalArrayFormattedOutput)(const Descriptor &descriptor,
    const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, void **scratchArea,
    std::size_t scratchBytes, const char *sourceFile, int sourceLine) {
  return BeginInternalArrayFormattedIO<Direction::Output>(descriptor, format,
      formatLength, formatDescriptor, scratchArea, scratchBytes, sourceFile,
      sourceLine);
}

Cookie IONAME(BeginInternalArrayFormattedInput)(const Descriptor &descriptor,
    const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, void **scratchArea,
    std::size_t scratchBytes, const char *sourceFile, int sourceLine) {
  return BeginInternalArrayFormattedIO<Direction::Input>(descriptor, format,
      formatLength, formatDescriptor, scratchArea, scratchBytes, sourceFile,
      sourceLine);
}

template <Direction DIR>
Cookie BeginInternalListIO(
    std::conditional_t<DIR == Direction::Input, const char, char> *internal,
    std::size_t internalLength, void ** /*scratchArea*/,
    std::size_t /*scratchBytes*/, const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalListIoStatementState<DIR>>{oom}(
      internal, internalLength, sourceFile, sourceLine)
              .release()
              ->ioStatementState();
}

Cookie IONAME(BeginInternalListOutput)(char *internal,
    std::size_t internalLength, void **scratchArea, std::size_t scratchBytes,
    const char *sourceFile, int sourceLine) {
  return BeginInternalListIO<Direction::Output>(internal, internalLength,
      scratchArea, scratchBytes, sourceFile, sourceLine);
}

Cookie IONAME(BeginInternalListInput)(const char *internal,
    std::size_t internalLength, void **scratchArea, std::size_t scratchBytes,
    const char *sourceFile, int sourceLine) {
  return BeginInternalListIO<Direction::Input>(internal, internalLength,
      scratchArea, scratchBytes, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginInternalFormattedIO(
    std::conditional_t<DIR == Direction::Input, const char, char> *internal,
    std::size_t internalLength, const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, void ** /*scratchArea*/,
    std::size_t /*scratchBytes*/, const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalFormattedIoStatementState<DIR>>{oom}(internal,
      internalLength, format, formatLength, formatDescriptor, sourceFile,
      sourceLine)
              .release()
              ->ioStatementState();
}

Cookie IONAME(BeginInternalFormattedOutput)(char *internal,
    std::size_t internalLength, const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, void **scratchArea,
    std::size_t scratchBytes, const char *sourceFile, int sourceLine) {
  return BeginInternalFormattedIO<Direction::Output>(internal, internalLength,
      format, formatLength, formatDescriptor, scratchArea, scratchBytes,
      sourceFile, sourceLine);
}

Cookie IONAME(BeginInternalFormattedInput)(const char *internal,
    std::size_t internalLength, const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, void **scratchArea,
    std::size_t scratchBytes, const char *sourceFile, int sourceLine) {
  return BeginInternalFormattedIO<Direction::Input>(internal, internalLength,
      format, formatLength, formatDescriptor, scratchArea, scratchBytes,
      sourceFile, sourceLine);
}

static Cookie NoopUnit(const Terminator &terminator, int unitNumber,
    enum Iostat iostat = IostatOk) {
  Cookie cookie{&New<NoopStatementState>{terminator}(
      terminator.sourceFileName(), terminator.sourceLine(), unitNumber)
                     .release()
                     ->ioStatementState()};
  if (iostat != IostatOk) {
    cookie->GetIoErrorHandler().SetPendingError(iostat);
  }
  return cookie;
}

static ExternalFileUnit *GetOrCreateUnit(int unitNumber, Direction direction,
    std::optional<bool> isUnformatted, const Terminator &terminator,
    Cookie &errorCookie) {
  if (ExternalFileUnit *
      unit{ExternalFileUnit::LookUpOrCreateAnonymous(
          unitNumber, direction, isUnformatted, terminator)}) {
    errorCookie = nullptr;
    return unit;
  } else {
    errorCookie = NoopUnit(terminator, unitNumber, IostatBadUnitNumber);
    return nullptr;
  }
}

template <Direction DIR, template <Direction> class STATE, typename... A>
Cookie BeginExternalListIO(
    int unitNumber, const char *sourceFile, int sourceLine, A &&...xs) {
  Terminator terminator{sourceFile, sourceLine};
  if (unitNumber == DefaultUnit) {
    unitNumber = DIR == Direction::Input ? 5 : 6;
  }
  Cookie errorCookie{nullptr};
  ExternalFileUnit *unit{GetOrCreateUnit(
      unitNumber, DIR, false /*!unformatted*/, terminator, errorCookie)};
  if (!unit) {
    return errorCookie;
  }
  if (!unit->isUnformatted.has_value()) {
    unit->isUnformatted = false;
  }
  Iostat iostat{IostatOk};
  if (*unit->isUnformatted) {
    iostat = IostatFormattedIoOnUnformattedUnit;
  }
  if (ChildIo * child{unit->GetChildIo()}) {
    if (iostat == IostatOk) {
      iostat = child->CheckFormattingAndDirection(false, DIR);
    }
    if (iostat == IostatOk) {
      return &child->BeginIoStatement<ChildListIoStatementState<DIR>>(
          *child, sourceFile, sourceLine);
    } else {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          iostat, nullptr /* no unit */, sourceFile, sourceLine);
    }
  } else {
    if (iostat == IostatOk && unit->access == Access::Direct) {
      iostat = IostatListIoOnDirectAccessUnit;
    }
    if (iostat == IostatOk) {
      iostat = unit->SetDirection(DIR);
    }
    if (iostat == IostatOk) {
      return &unit->BeginIoStatement<STATE<DIR>>(
          terminator, std::forward<A>(xs)..., *unit, sourceFile, sourceLine);
    } else {
      return &unit->BeginIoStatement<ErroneousIoStatementState>(
          terminator, iostat, unit, sourceFile, sourceLine);
    }
  }
}

Cookie IONAME(BeginExternalListOutput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalListIO<Direction::Output, ExternalListIoStatementState>(
      unitNumber, sourceFile, sourceLine);
}

Cookie IONAME(BeginExternalListInput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalListIO<Direction::Input, ExternalListIoStatementState>(
      unitNumber, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginExternalFormattedIO(const char *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, ExternalUnit unitNumber,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (unitNumber == DefaultUnit) {
    unitNumber = DIR == Direction::Input ? 5 : 6;
  }
  Cookie errorCookie{nullptr};
  ExternalFileUnit *unit{GetOrCreateUnit(
      unitNumber, DIR, false /*!unformatted*/, terminator, errorCookie)};
  if (!unit) {
    return errorCookie;
  }
  Iostat iostat{IostatOk};
  if (!unit->isUnformatted.has_value()) {
    unit->isUnformatted = false;
  }
  if (*unit->isUnformatted) {
    iostat = IostatFormattedIoOnUnformattedUnit;
  }
  if (ChildIo * child{unit->GetChildIo()}) {
    if (iostat == IostatOk) {
      iostat = child->CheckFormattingAndDirection(false, DIR);
    }
    if (iostat == IostatOk) {
      return &child->BeginIoStatement<ChildFormattedIoStatementState<DIR>>(
          *child, format, formatLength, formatDescriptor, sourceFile,
          sourceLine);
    } else {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          iostat, nullptr /* no unit */, sourceFile, sourceLine);
    }
  } else {
    if (iostat == IostatOk) {
      iostat = unit->SetDirection(DIR);
    }
    if (iostat == IostatOk) {
      return &unit->BeginIoStatement<ExternalFormattedIoStatementState<DIR>>(
          terminator, *unit, format, formatLength, formatDescriptor, sourceFile,
          sourceLine);
    } else {
      return &unit->BeginIoStatement<ErroneousIoStatementState>(
          terminator, iostat, unit, sourceFile, sourceLine);
    }
  }
}

Cookie IONAME(BeginExternalFormattedOutput)(const char *format,
    std::size_t formatLength, const Descriptor *formatDescriptor,
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalFormattedIO<Direction::Output>(format, formatLength,
      formatDescriptor, unitNumber, sourceFile, sourceLine);
}

Cookie IONAME(BeginExternalFormattedInput)(const char *format,
    std::size_t formatLength, const Descriptor *formatDescriptor,
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalFormattedIO<Direction::Input>(format, formatLength,
      formatDescriptor, unitNumber, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginUnformattedIO(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Cookie errorCookie{nullptr};
  ExternalFileUnit *unit{GetOrCreateUnit(
      unitNumber, DIR, true /*unformatted*/, terminator, errorCookie)};
  if (!unit) {
    return errorCookie;
  }
  Iostat iostat{IostatOk};
  if (!unit->isUnformatted.has_value()) {
    unit->isUnformatted = true;
  }
  if (!*unit->isUnformatted) {
    iostat = IostatUnformattedIoOnFormattedUnit;
  }
  if (ChildIo * child{unit->GetChildIo()}) {
    if (iostat == IostatOk) {
      iostat = child->CheckFormattingAndDirection(true, DIR);
    }
    if (iostat == IostatOk) {
      return &child->BeginIoStatement<ChildUnformattedIoStatementState<DIR>>(
          *child, sourceFile, sourceLine);
    } else {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          iostat, nullptr /* no unit */, sourceFile, sourceLine);
    }
  } else {
    if (iostat == IostatOk) {
      iostat = unit->SetDirection(DIR);
    }
    if (iostat == IostatOk) {
      IoStatementState &io{
          unit->BeginIoStatement<ExternalUnformattedIoStatementState<DIR>>(
              terminator, *unit, sourceFile, sourceLine)};
      if constexpr (DIR == Direction::Output) {
        if (unit->access == Access::Sequential) {
          // Create space for (sub)record header to be completed by
          // ExternalFileUnit::AdvanceRecord()
          unit->recordLength.reset(); // in case of prior BACKSPACE
          io.Emit("\0\0\0\0", 4); // placeholder for record length header
        }
      }
      return &io;
    } else {
      return &unit->BeginIoStatement<ErroneousIoStatementState>(
          terminator, iostat, unit, sourceFile, sourceLine);
    }
  }
}

Cookie IONAME(BeginUnformattedOutput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginUnformattedIO<Direction::Output>(
      unitNumber, sourceFile, sourceLine);
}

Cookie IONAME(BeginUnformattedInput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginUnformattedIO<Direction::Input>(
      unitNumber, sourceFile, sourceLine);
}

Cookie IONAME(BeginOpenUnit)( // OPEN(without NEWUNIT=)
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  bool wasExtant{false};
  if (ExternalFileUnit *
      unit{ExternalFileUnit::LookUpOrCreate(
          unitNumber, terminator, wasExtant)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          IostatBadOpOnChildUnit, nullptr /* no unit */, sourceFile,
          sourceLine);
    } else {
      return &unit->BeginIoStatement<OpenStatementState>(
          terminator, *unit, wasExtant, sourceFile, sourceLine);
    }
  } else {
    return NoopUnit(terminator, unitNumber, IostatBadUnitNumber);
  }
}

Cookie IONAME(BeginOpenNewUnit)( // OPEN(NEWUNIT=j)
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{
      ExternalFileUnit::NewUnit(terminator, false /*not child I/O*/)};
  return &unit.BeginIoStatement<OpenStatementState>(
      terminator, unit, false /*was an existing file*/, sourceFile, sourceLine);
}

Cookie IONAME(BeginWait)(ExternalUnit unitNumber, AsynchronousId id,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(unitNumber)}) {
    if (unit->Wait(id)) {
      return &unit->BeginIoStatement<ExternalMiscIoStatementState>(terminator,
          *unit, ExternalMiscIoStatementState::Wait, sourceFile, sourceLine);
    } else {
      return &unit->BeginIoStatement<ErroneousIoStatementState>(
          terminator, IostatBadWaitId, unit, sourceFile, sourceLine);
    }
  } else {
    return NoopUnit(
        terminator, unitNumber, id == 0 ? IostatOk : IostatBadWaitUnit);
  }
}
Cookie IONAME(BeginWaitAll)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return IONAME(BeginWait)(unitNumber, 0 /*no ID=*/, sourceFile, sourceLine);
}

Cookie IONAME(BeginClose)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(unitNumber)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          IostatBadOpOnChildUnit, nullptr /* no unit */, sourceFile,
          sourceLine);
    }
  }
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUpForClose(unitNumber)}) {
    return &unit->BeginIoStatement<CloseStatementState>(
        terminator, *unit, sourceFile, sourceLine);
  } else {
    // CLOSE(UNIT=bad unit) is just a no-op
    return NoopUnit(terminator, unitNumber);
  }
}

Cookie IONAME(BeginFlush)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(unitNumber)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<ExternalMiscIoStatementState>(
          *unit, ExternalMiscIoStatementState::Flush, sourceFile, sourceLine);
    } else {
      return &unit->BeginIoStatement<ExternalMiscIoStatementState>(terminator,
          *unit, ExternalMiscIoStatementState::Flush, sourceFile, sourceLine);
    }
  } else {
    // FLUSH(UNIT=bad unit) is an error; an unconnected unit is a no-op
    return NoopUnit(terminator, unitNumber,
        unitNumber >= 0 ? IostatOk : IostatBadFlushUnit);
  }
}

Cookie IONAME(BeginBackspace)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(unitNumber)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          IostatBadOpOnChildUnit, nullptr /* no unit */, sourceFile,
          sourceLine);
    } else {
      return &unit->BeginIoStatement<ExternalMiscIoStatementState>(terminator,
          *unit, ExternalMiscIoStatementState::Backspace, sourceFile,
          sourceLine);
    }
  } else {
    return NoopUnit(terminator, unitNumber, IostatBadBackspaceUnit);
  }
}

Cookie IONAME(BeginEndfile)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Cookie errorCookie{nullptr};
  if (ExternalFileUnit *
      unit{GetOrCreateUnit(unitNumber, Direction::Output, std::nullopt,
          terminator, errorCookie)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          IostatBadOpOnChildUnit, nullptr /* no unit */, sourceFile,
          sourceLine);
    } else {
      return &unit->BeginIoStatement<ExternalMiscIoStatementState>(terminator,
          *unit, ExternalMiscIoStatementState::Endfile, sourceFile, sourceLine);
    }
  } else {
    return errorCookie;
  }
}

Cookie IONAME(BeginRewind)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  Cookie errorCookie{nullptr};
  if (ExternalFileUnit *
      unit{GetOrCreateUnit(unitNumber, Direction::Input, std::nullopt,
          terminator, errorCookie)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          IostatBadOpOnChildUnit, nullptr /* no unit */, sourceFile,
          sourceLine);
    } else {
      return &unit->BeginIoStatement<ExternalMiscIoStatementState>(terminator,
          *unit, ExternalMiscIoStatementState::Rewind, sourceFile, sourceLine);
    }
  } else {
    return errorCookie;
  }
}

Cookie IONAME(BeginInquireUnit)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(unitNumber)}) {
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<InquireUnitState>(
          *unit, sourceFile, sourceLine);
    } else {
      return &unit->BeginIoStatement<InquireUnitState>(
          terminator, *unit, sourceFile, sourceLine);
    }
  } else {
    // INQUIRE(UNIT=unrecognized unit)
    return &New<InquireNoUnitState>{terminator}(
        sourceFile, sourceLine, unitNumber)
                .release()
                ->ioStatementState();
  }
}

Cookie IONAME(BeginInquireFile)(const char *path, std::size_t pathLength,
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  auto trimmed{SaveDefaultCharacter(
      path, TrimTrailingSpaces(path, pathLength), terminator)};
  if (ExternalFileUnit *
      unit{ExternalFileUnit::LookUp(
          trimmed.get(), std::strlen(trimmed.get()))}) {
    // INQUIRE(FILE=) to a connected unit
    if (ChildIo * child{unit->GetChildIo()}) {
      return &child->BeginIoStatement<InquireUnitState>(
          *unit, sourceFile, sourceLine);
    } else {
      return &unit->BeginIoStatement<InquireUnitState>(
          terminator, *unit, sourceFile, sourceLine);
    }
  } else {
    return &New<InquireUnconnectedFileState>{terminator}(
        std::move(trimmed), sourceFile, sourceLine)
                .release()
                ->ioStatementState();
  }
}

Cookie IONAME(BeginInquireIoLength)(const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InquireIOLengthState>{oom}(sourceFile, sourceLine)
              .release()
              ->ioStatementState();
}

// Control list items

void IONAME(EnableHandlers)(Cookie cookie, bool hasIoStat, bool hasErr,
    bool hasEnd, bool hasEor, bool hasIoMsg) {
  IoErrorHandler &handler{cookie->GetIoErrorHandler()};
  if (hasIoStat) {
    handler.HasIoStat();
  }
  if (hasErr) {
    handler.HasErrLabel();
  }
  if (hasEnd) {
    handler.HasEndLabel();
  }
  if (hasEor) {
    handler.HasEorLabel();
  }
  if (hasIoMsg) {
    handler.HasIoMsg();
  }
}

static bool YesOrNo(const char *keyword, std::size_t length, const char *what,
    IoErrorHandler &handler) {
  static const char *keywords[]{"YES", "NO", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    return true;
  case 1:
    return false;
  default:
    handler.SignalError(IostatErrorInKeyword, "Invalid %s='%.*s'", what,
        static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetAdvance)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  bool nonAdvancing{!YesOrNo(keyword, length, "ADVANCE", handler)};
  if (nonAdvancing && io.GetConnectionState().access == Access::Direct) {
    handler.SignalError("Non-advancing I/O attempted on direct access file");
  } else {
    auto *unit{io.GetExternalFileUnit()};
    if (unit && unit->GetChildIo()) {
      // ADVANCE= is ignored for child I/O (12.6.4.8.3 p3)
    } else {
      io.mutableModes().nonAdvancing = nonAdvancing;
    }
  }
  return !handler.InError();
}

bool IONAME(SetBlank)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  static const char *keywords[]{"NULL", "ZERO", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    io.mutableModes().editingFlags &= ~blankZero;
    return true;
  case 1:
    io.mutableModes().editingFlags |= blankZero;
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid BLANK='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetDecimal)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  static const char *keywords[]{"COMMA", "POINT", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    io.mutableModes().editingFlags |= decimalComma;
    return true;
  case 1:
    io.mutableModes().editingFlags &= ~decimalComma;
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid DECIMAL='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetDelim)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  static const char *keywords[]{"APOSTROPHE", "QUOTE", "NONE", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    io.mutableModes().delim = '\'';
    return true;
  case 1:
    io.mutableModes().delim = '"';
    return true;
  case 2:
    io.mutableModes().delim = '\0';
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid DELIM='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetPad)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  io.mutableModes().pad = YesOrNo(keyword, length, "PAD", handler);
  return !handler.InError();
}

bool IONAME(SetPos)(Cookie cookie, std::int64_t pos) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (auto *unit{io.GetExternalFileUnit()}) {
    return unit->SetStreamPos(pos, handler);
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    handler.Crash("SetPos() called on internal unit");
  }
  return false;
}

bool IONAME(SetRec)(Cookie cookie, std::int64_t rec) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (auto *unit{io.GetExternalFileUnit()}) {
    if (unit->GetChildIo()) {
      handler.SignalError(
          IostatBadOpOnChildUnit, "REC= specifier on child I/O");
    } else {
      unit->SetDirectRec(rec, handler);
    }
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    handler.Crash("SetRec() called on internal unit");
  }
  return true;
}

bool IONAME(SetRound)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  static const char *keywords[]{"UP", "DOWN", "ZERO", "NEAREST", "COMPATIBLE",
      "PROCESSOR_DEFINED", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    io.mutableModes().round = decimal::RoundUp;
    return true;
  case 1:
    io.mutableModes().round = decimal::RoundDown;
    return true;
  case 2:
    io.mutableModes().round = decimal::RoundToZero;
    return true;
  case 3:
    io.mutableModes().round = decimal::RoundNearest;
    return true;
  case 4:
    io.mutableModes().round = decimal::RoundCompatible;
    return true;
  case 5:
    io.mutableModes().round = executionEnvironment.defaultOutputRoundingMode;
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid ROUND='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetSign)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  static const char *keywords[]{
      "PLUS", "SUPPRESS", "PROCESSOR_DEFINED", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    io.mutableModes().editingFlags |= signPlus;
    return true;
  case 1:
  case 2: // processor default is SS
    io.mutableModes().editingFlags &= ~signPlus;
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid SIGN='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetAccess)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetAccess() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetAccess() called after GetNewUnit() for an OPEN statement");
  }
  static const char *keywords[]{
      "SEQUENTIAL", "DIRECT", "STREAM", "APPEND", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    open->set_access(Access::Sequential);
    break;
  case 1:
    open->set_access(Access::Direct);
    break;
  case 2:
    open->set_access(Access::Stream);
    break;
  case 3: // Sun Fortran extension ACCESS=APPEND: treat as if POSITION=APPEND
    open->set_position(Position::Append);
    break;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid ACCESS='%.*s'",
        static_cast<int>(length), keyword);
  }
  return true;
}

bool IONAME(SetAction)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetAction() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetAction() called after GetNewUnit() for an OPEN statement");
  }
  std::optional<Action> action;
  static const char *keywords[]{"READ", "WRITE", "READWRITE", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    action = Action::Read;
    break;
  case 1:
    action = Action::Write;
    break;
  case 2:
    action = Action::ReadWrite;
    break;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid ACTION='%.*s'",
        static_cast<int>(length), keyword);
    return false;
  }
  RUNTIME_CHECK(io.GetIoErrorHandler(), action.has_value());
  if (open->wasExtant()) {
    if ((*action != Action::Write) != open->unit().mayRead() ||
        (*action != Action::Read) != open->unit().mayWrite()) {
      open->SignalError("ACTION= may not be changed on an open unit");
    }
  }
  open->set_action(*action);
  return true;
}

bool IONAME(SetAsynchronous)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  bool isYes{YesOrNo(keyword, length, "ASYNCHRONOUS", handler)};
  if (auto *open{io.get_if<OpenStatementState>()}) {
    if (open->completedOperation()) {
      handler.Crash(
          "SetAsynchronous() called after GetNewUnit() for an OPEN statement");
    }
    open->unit().set_mayAsynchronous(isYes);
  } else if (auto *ext{io.get_if<ExternalIoStatementBase>()}) {
    if (isYes) {
      if (ext->unit().mayAsynchronous()) {
        ext->SetAsynchronous();
      } else {
        handler.SignalError(IostatBadAsynchronous);
      }
    }
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    handler.Crash("SetAsynchronous() called when not in an OPEN or external "
                  "I/O statement");
  }
  return !handler.InError();
}

bool IONAME(SetCarriagecontrol)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetCarriageControl() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetCarriageControl() called after GetNewUnit() for an OPEN statement");
  }
  static const char *keywords[]{"LIST", "FORTRAN", "NONE", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    return true;
  case 1:
  case 2:
    open->SignalError(IostatErrorInKeyword,
        "Unimplemented CARRIAGECONTROL='%.*s'", static_cast<int>(length),
        keyword);
    return false;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid CARRIAGECONTROL='%.*s'",
        static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetConvert)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetConvert() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetConvert() called after GetNewUnit() for an OPEN statement");
  }
  if (auto convert{GetConvertFromString(keyword, length)}) {
    open->set_convert(*convert);
    return true;
  } else {
    open->SignalError(IostatErrorInKeyword, "Invalid CONVERT='%.*s'",
        static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetEncoding)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetEncoding() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetEncoding() called after GetNewUnit() for an OPEN statement");
  }
  bool isUTF8{false};
  static const char *keywords[]{"UTF-8", "DEFAULT", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    isUTF8 = true;
    break;
  case 1:
    isUTF8 = false;
    break;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid ENCODING='%.*s'",
        static_cast<int>(length), keyword);
  }
  if (isUTF8 != open->unit().isUTF8) {
    if (open->wasExtant()) {
      open->SignalError("ENCODING= may not be changed on an open unit");
    }
    open->unit().isUTF8 = isUTF8;
  }
  return true;
}

bool IONAME(SetForm)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetForm() called when not in an OPEN statement");
    }
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetForm() called after GetNewUnit() for an OPEN statement");
  }
  static const char *keywords[]{"FORMATTED", "UNFORMATTED", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    open->set_isUnformatted(false);
    break;
  case 1:
    open->set_isUnformatted(true);
    break;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid FORM='%.*s'",
        static_cast<int>(length), keyword);
  }
  return true;
}

bool IONAME(SetPosition)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetPosition() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetPosition() called after GetNewUnit() for an OPEN statement");
  }
  static const char *positions[]{"ASIS", "REWIND", "APPEND", nullptr};
  switch (IdentifyValue(keyword, length, positions)) {
  case 0:
    open->set_position(Position::AsIs);
    return true;
  case 1:
    open->set_position(Position::Rewind);
    return true;
  case 2:
    open->set_position(Position::Append);
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid POSITION='%.*s'", static_cast<int>(length), keyword);
  }
  return true;
}

bool IONAME(SetRecl)(Cookie cookie, std::size_t n) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "SetRecl() called when not in an OPEN statement");
    }
    return false;
  } else if (open->completedOperation()) {
    io.GetIoErrorHandler().Crash(
        "SetRecl() called after GetNewUnit() for an OPEN statement");
  }
  if (n <= 0) {
    io.GetIoErrorHandler().SignalError("RECL= must be greater than zero");
    return false;
  } else if (open->wasExtant() &&
      open->unit().openRecl.value_or(0) != static_cast<std::int64_t>(n)) {
    open->SignalError("RECL= may not be changed for an open unit");
    return false;
  } else {
    open->unit().openRecl = n;
    return true;
  }
}

bool IONAME(SetStatus)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  if (auto *open{io.get_if<OpenStatementState>()}) {
    if (open->completedOperation()) {
      io.GetIoErrorHandler().Crash(
          "SetStatus() called after GetNewUnit() for an OPEN statement");
    }
    static const char *statuses[]{
        "OLD", "NEW", "SCRATCH", "REPLACE", "UNKNOWN", nullptr};
    switch (IdentifyValue(keyword, length, statuses)) {
    case 0:
      open->set_status(OpenStatus::Old);
      return true;
    case 1:
      open->set_status(OpenStatus::New);
      return true;
    case 2:
      open->set_status(OpenStatus::Scratch);
      return true;
    case 3:
      open->set_status(OpenStatus::Replace);
      return true;
    case 4:
      open->set_status(OpenStatus::Unknown);
      return true;
    default:
      io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
          "Invalid STATUS='%.*s'", static_cast<int>(length), keyword);
    }
    return false;
  }
  if (auto *close{io.get_if<CloseStatementState>()}) {
    static const char *statuses[]{"KEEP", "DELETE", nullptr};
    switch (IdentifyValue(keyword, length, statuses)) {
    case 0:
      close->set_status(CloseStatus::Keep);
      return true;
    case 1:
      close->set_status(CloseStatus::Delete);
      return true;
    default:
      io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
          "Invalid STATUS='%.*s'", static_cast<int>(length), keyword);
    }
    return false;
  }
  if (io.get_if<NoopStatementState>() ||
      io.get_if<ErroneousIoStatementState>()) {
    return true; // don't bother validating STATUS= in a no-op CLOSE
  }
  io.GetIoErrorHandler().Crash(
      "SetStatus() called when not in an OPEN or CLOSE statement");
}

bool IONAME(SetFile)(Cookie cookie, const char *path, std::size_t chars) {
  IoStatementState &io{*cookie};
  if (auto *open{io.get_if<OpenStatementState>()}) {
    if (open->completedOperation()) {
      io.GetIoErrorHandler().Crash(
          "SetFile() called after GetNewUnit() for an OPEN statement");
    }
    open->set_path(path, chars);
    return true;
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "SetFile() called when not in an OPEN statement");
  }
  return false;
}

bool IONAME(GetNewUnit)(Cookie cookie, int &unit, int kind) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    if (!io.get_if<ErroneousIoStatementState>()) {
      io.GetIoErrorHandler().Crash(
          "GetNewUnit() called when not in an OPEN statement");
    }
    return false;
  } else if (!open->InError()) {
    open->CompleteOperation();
  }
  if (open->InError()) {
    // A failed OPEN(NEWUNIT=n) does not modify 'n'
    return false;
  }
  std::int64_t result{open->unit().unitNumber()};
  if (!SetInteger(unit, kind, result)) {
    open->SignalError("GetNewUnit(): bad INTEGER kind(%d) or out-of-range "
                      "value(%jd) for result",
        kind, static_cast<std::intmax_t>(result));
  }
  return true;
}

// Data transfers

bool IONAME(OutputDescriptor)(Cookie cookie, const Descriptor &descriptor) {
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputDescriptor)(Cookie cookie, const Descriptor &descriptor) {
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputUnformattedBlock)(Cookie cookie, const char *x,
    std::size_t length, std::size_t elementBytes) {
  IoStatementState &io{*cookie};
  if (auto *unf{io.get_if<
          ExternalUnformattedIoStatementState<Direction::Output>>()}) {
    return unf->Emit(x, length, elementBytes);
  } else if (auto *inq{io.get_if<InquireIOLengthState>()}) {
    return inq->Emit(x, length, elementBytes);
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    io.GetIoErrorHandler().Crash("OutputUnformattedBlock() called for an I/O "
                                 "statement that is not unformatted output");
  }
  return false;
}

bool IONAME(InputUnformattedBlock)(
    Cookie cookie, char *x, std::size_t length, std::size_t elementBytes) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  io.BeginReadingRecord();
  if (handler.InError()) {
    return false;
  }
  if (auto *unf{
          io.get_if<ExternalUnformattedIoStatementState<Direction::Input>>()}) {
    return unf->Receive(x, length, elementBytes);
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    handler.Crash("InputUnformattedBlock() called for an I/O statement that is "
                  "not unformatted input");
  }
  return false;
}

bool IONAME(OutputInteger8)(Cookie cookie, std::int8_t n) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputInteger8")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, 1, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputInteger16)(Cookie cookie, std::int16_t n) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputInteger16")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, 2, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputInteger32)(Cookie cookie, std::int32_t n) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputInteger32")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, 4, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputInteger64)(Cookie cookie, std::int64_t n) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputInteger64")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, 8, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

#ifdef __SIZEOF_INT128__
bool IONAME(OutputInteger128)(Cookie cookie, common::int128_t n) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputInteger128")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, 16, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}
#endif

bool IONAME(InputInteger)(Cookie cookie, std::int64_t &n, int kind) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputInteger")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, kind, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputReal32)(Cookie cookie, float x) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputReal32")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 4, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputReal64)(Cookie cookie, double x) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputReal64")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 8, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputReal32)(Cookie cookie, float &x) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputReal32")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 4, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(InputReal64)(Cookie cookie, double &x) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputReal64")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 8, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputComplex32)(Cookie cookie, float r, float i) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputComplex32")) {
    return false;
  }
  float z[2]{r, i};
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 4, reinterpret_cast<void *>(&z), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputComplex64)(Cookie cookie, double r, double i) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputComplex64")) {
    return false;
  }
  double z[2]{r, i};
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 8, reinterpret_cast<void *>(&z), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputComplex32)(Cookie cookie, float z[2]) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputComplex32")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 4, reinterpret_cast<void *>(z), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(InputComplex64)(Cookie cookie, double z[2]) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputComplex64")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 8, reinterpret_cast<void *>(z), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputCharacter)(
    Cookie cookie, const char *x, std::size_t length, int kind) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputCharacter")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      kind, length, reinterpret_cast<void *>(const_cast<char *>(x)), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  return IONAME(OutputCharacter(cookie, x, length, 1));
}

bool IONAME(InputCharacter)(
    Cookie cookie, char *x, std::size_t length, int kind) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputCharacter")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(kind, length, reinterpret_cast<void *>(x), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(InputAscii)(Cookie cookie, char *x, std::size_t length) {
  return IONAME(InputCharacter)(cookie, x, length, 1);
}

bool IONAME(OutputLogical)(Cookie cookie, bool truth) {
  if (!cookie->CheckFormattedStmtType<Direction::Output>("OutputLogical")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Logical, sizeof truth, reinterpret_cast<void *>(&truth), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputLogical)(Cookie cookie, bool &truth) {
  if (!cookie->CheckFormattedStmtType<Direction::Input>("InputLogical")) {
    return false;
  }
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Logical, sizeof truth, reinterpret_cast<void *>(&truth), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

template <Direction DIR>
static bool DoDerivedTypeIo(Cookie cookie, const Descriptor &descriptor,
    void (*procedure)(), bool isPolymorphic, const char *which) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (handler.InError()) {
    return false;
  }
  const DescriptorAddendum *addendum{descriptor.Addendum()};
  const typeInfo::DerivedType *type{
      addendum ? addendum->derivedType() : nullptr};
  RUNTIME_CHECK(handler, type != nullptr);
  if (!procedure) {
    if constexpr (DIR == Direction::Output) {
      return IONAME(OutputDescriptor)(cookie, descriptor);
    } else {
      return IONAME(InputDescriptor)(cookie, descriptor);
    }
  }
  if (!io.get_if<IoDirectionState<DIR>>()) {
    handler.Crash("%s called for I/O statement that is not %s", which,
        DIR == Direction::Output ? "output" : "input");
  }
  std::uint8_t isArgDesc{isPolymorphic};
  if (io.get_if<FormattedIoStatementState<DIR>>()) {
    if (std::optional<bool> wasDefined{
            descr::DefinedFormattedIo(io, descriptor, *type,
                typeInfo::SpecialBinding{DIR == Direction::Output
                        ? typeInfo::SpecialBinding::Which::WriteFormatted
                        : typeInfo::SpecialBinding::Which::ReadFormatted,
                    procedure, isArgDesc})}) {
      return *wasDefined;
    }
    return descr::DefaultComponentwiseIO<DIR>(io, descriptor, *type);
  } else { // unformatted
    return descr::DefinedUnformattedIo(io, descriptor, *type,
        typeInfo::SpecialBinding{DIR == Direction::Output
                ? typeInfo::SpecialBinding::Which::WriteUnformatted
                : typeInfo::SpecialBinding::Which::ReadUnformatted,
            procedure, isArgDesc});
  }
}

bool IONAME(OutputDerivedType)(Cookie cookie, const Descriptor &descriptor,
    void (*procedure)(), bool isPolymorphic) {
  return DoDerivedTypeIo<Direction::Output>(
      cookie, descriptor, procedure, isPolymorphic, "OutputDerivedType");
}

bool IONAME(InputDerivedType)(Cookie cookie, const Descriptor &descriptor,
    void (*procedure)(), bool isPolymorphic) {
  return DoDerivedTypeIo<Direction::Output>(
      cookie, descriptor, procedure, isPolymorphic, "InputDerivedType");
}

std::size_t IONAME(GetSize)(Cookie cookie) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (!handler.InError()) {
    io.CompleteOperation();
  }
  if (const auto *formatted{
          io.get_if<FormattedIoStatementState<Direction::Input>>()}) {
    return formatted->GetEditDescriptorChars();
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    handler.Crash("GetIoSize() called for an I/O statement that is not a "
                  "formatted READ()");
  }
  return 0;
}

std::size_t IONAME(GetIoLength)(Cookie cookie) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (!handler.InError()) {
    io.CompleteOperation();
  }
  if (const auto *inq{io.get_if<InquireIOLengthState>()}) {
    return inq->bytes();
  } else if (!io.get_if<ErroneousIoStatementState>()) {
    handler.Crash("GetIoLength() called for an I/O statement that is not "
                  "INQUIRE(IOLENGTH=)");
  }
  return 0;
}

void IONAME(GetIoMsg)(Cookie cookie, char *msg, std::size_t length) {
  IoStatementState &io{*cookie};
  IoErrorHandler &handler{io.GetIoErrorHandler()};
  if (!handler.InError()) {
    io.CompleteOperation();
  }
  if (handler.InError()) { // leave "msg" alone when no error
    handler.GetIoMsg(msg, length);
  }
}

bool IONAME(InquireCharacter)(Cookie cookie, InquiryKeywordHash inquiry,
    char *result, std::size_t length) {
  IoStatementState &io{*cookie};
  return io.Inquire(inquiry, result, length);
}

bool IONAME(InquireLogical)(
    Cookie cookie, InquiryKeywordHash inquiry, bool &result) {
  IoStatementState &io{*cookie};
  return io.Inquire(inquiry, result);
}

bool IONAME(InquirePendingId)(Cookie cookie, std::int64_t id, bool &result) {
  IoStatementState &io{*cookie};
  return io.Inquire(HashInquiryKeyword("PENDING"), id, result);
}

bool IONAME(InquireInteger64)(
    Cookie cookie, InquiryKeywordHash inquiry, std::int64_t &result, int kind) {
  IoStatementState &io{*cookie};
  std::int64_t n;
  if (io.Inquire(inquiry, n)) {
    if (SetInteger(result, kind, n)) {
      return true;
    }
    io.GetIoErrorHandler().SignalError(
        "InquireInteger64(): bad INTEGER kind(%d) or out-of-range "
        "value(%jd) "
        "for result",
        kind, static_cast<std::intmax_t>(n));
  }
  return false;
}

enum Iostat IONAME(EndIoStatement)(Cookie cookie) {
  IoStatementState &io{*cookie};
  return static_cast<enum Iostat>(io.EndIoStatement());
}

template <typename INT>
static enum Iostat CheckUnitNumberInRangeImpl(INT unit, bool handleError,
    char *ioMsg, std::size_t ioMsgLength, const char *sourceFile,
    int sourceLine) {
  static_assert(sizeof(INT) >= sizeof(ExternalUnit),
      "only intended to be used when the INT to ExternalUnit conversion is "
      "narrowing");
  if (unit != static_cast<ExternalUnit>(unit)) {
    Terminator oom{sourceFile, sourceLine};
    IoErrorHandler errorHandler{oom};
    if (handleError) {
      errorHandler.HasIoStat();
      if (ioMsg) {
        errorHandler.HasIoMsg();
      }
    }
    // Only provide the bad unit number in the message if SignalError can print
    // it accurately. Otherwise, the generic IostatUnitOverflow message will be
    // used.
    if (static_cast<std::intmax_t>(unit) == unit) {
      errorHandler.SignalError(IostatUnitOverflow,
          "UNIT number %jd is out of range", static_cast<std::intmax_t>(unit));
    } else {
      errorHandler.SignalError(IostatUnitOverflow);
    }
    if (ioMsg) {
      errorHandler.GetIoMsg(ioMsg, ioMsgLength);
    }
    return static_cast<enum Iostat>(errorHandler.GetIoStat());
  }
  return IostatOk;
}

enum Iostat IONAME(CheckUnitNumberInRange64)(std::int64_t unit,
    bool handleError, char *ioMsg, std::size_t ioMsgLength,
    const char *sourceFile, int sourceLine) {
  return CheckUnitNumberInRangeImpl(
      unit, handleError, ioMsg, ioMsgLength, sourceFile, sourceLine);
}

#ifdef __SIZEOF_INT128__
enum Iostat IONAME(CheckUnitNumberInRange128)(common::int128_t unit,
    bool handleError, char *ioMsg, std::size_t ioMsgLength,
    const char *sourceFile, int sourceLine) {
  return CheckUnitNumberInRangeImpl(
      unit, handleError, ioMsg, ioMsgLength, sourceFile, sourceLine);
}
#endif

} // namespace Fortran::runtime::io
