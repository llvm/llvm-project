//===-- runtime/io-api.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the I/O statement API

#include "io-api.h"
#include "edit-input.h"
#include "edit-output.h"
#include "environment.h"
#include "format.h"
#include "io-stmt.h"
#include "memory.h"
#include "terminator.h"
#include "tools.h"
#include "unit.h"
#include <cstdlib>
#include <memory>

namespace Fortran::runtime::io {

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
    const char *format, std::size_t formatLength, void ** /*scratchArea*/,
    std::size_t /*scratchBytes*/, const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalFormattedIoStatementState<DIR>>{oom}(
      descriptor, format, formatLength, sourceFile, sourceLine)
              .release()
              ->ioStatementState();
}

Cookie IONAME(BeginInternalArrayFormattedOutput)(const Descriptor &descriptor,
    const char *format, std::size_t formatLength, void **scratchArea,
    std::size_t scratchBytes, const char *sourceFile, int sourceLine) {
  return BeginInternalArrayFormattedIO<Direction::Output>(descriptor, format,
      formatLength, scratchArea, scratchBytes, sourceFile, sourceLine);
}

Cookie IONAME(BeginInternalArrayFormattedInput)(const Descriptor &descriptor,
    const char *format, std::size_t formatLength, void **scratchArea,
    std::size_t scratchBytes, const char *sourceFile, int sourceLine) {
  return BeginInternalArrayFormattedIO<Direction::Input>(descriptor, format,
      formatLength, scratchArea, scratchBytes, sourceFile, sourceLine);
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
    void ** /*scratchArea*/, std::size_t /*scratchBytes*/,
    const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  return &New<InternalFormattedIoStatementState<DIR>>{oom}(
      internal, internalLength, format, formatLength, sourceFile, sourceLine)
              .release()
              ->ioStatementState();
}

Cookie IONAME(BeginInternalFormattedOutput)(char *internal,
    std::size_t internalLength, const char *format, std::size_t formatLength,
    void **scratchArea, std::size_t scratchBytes, const char *sourceFile,
    int sourceLine) {
  return BeginInternalFormattedIO<Direction::Output>(internal, internalLength,
      format, formatLength, scratchArea, scratchBytes, sourceFile, sourceLine);
}

Cookie IONAME(BeginInternalFormattedInput)(const char *internal,
    std::size_t internalLength, const char *format, std::size_t formatLength,
    void **scratchArea, std::size_t scratchBytes, const char *sourceFile,
    int sourceLine) {
  return BeginInternalFormattedIO<Direction::Input>(internal, internalLength,
      format, formatLength, scratchArea, scratchBytes, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginExternalListIO(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (unitNumber == DefaultUnit) {
    unitNumber = DIR == Direction::Input ? 5 : 6;
  }
  ExternalFileUnit &unit{ExternalFileUnit::LookUpOrCreateAnonymous(
      unitNumber, DIR, false /*formatted*/, terminator)};
  if (unit.access == Access::Direct) {
    terminator.Crash("List-directed I/O attempted on direct access file");
    return nullptr;
  }
  if (unit.isUnformatted) {
    terminator.Crash("List-directed I/O attempted on unformatted file");
    return nullptr;
  }
  IoErrorHandler handler{terminator};
  unit.SetDirection(DIR, handler);
  IoStatementState &io{unit.BeginIoStatement<ExternalListIoStatementState<DIR>>(
      unit, sourceFile, sourceLine)};
  if constexpr (DIR == Direction::Input) {
    unit.BeginReadingRecord(handler);
  }
  return &io;
}

Cookie IONAME(BeginExternalListOutput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalListIO<Direction::Output>(
      unitNumber, sourceFile, sourceLine);
}

Cookie IONAME(BeginExternalListInput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalListIO<Direction::Input>(
      unitNumber, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginExternalFormattedIO(const char *format, std::size_t formatLength,
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (unitNumber == DefaultUnit) {
    unitNumber = DIR == Direction::Input ? 5 : 6;
  }
  ExternalFileUnit &unit{ExternalFileUnit::LookUpOrCreateAnonymous(
      unitNumber, DIR, false /*formatted*/, terminator)};
  if (unit.isUnformatted) {
    terminator.Crash("Formatted I/O attempted on unformatted file");
    return nullptr;
  }
  IoErrorHandler handler{terminator};
  unit.SetDirection(DIR, handler);
  IoStatementState &io{
      unit.BeginIoStatement<ExternalFormattedIoStatementState<DIR>>(
          unit, format, formatLength, sourceFile, sourceLine)};
  if constexpr (DIR == Direction::Input) {
    unit.BeginReadingRecord(handler);
  }
  return &io;
}

Cookie IONAME(BeginExternalFormattedOutput)(const char *format,
    std::size_t formatLength, ExternalUnit unitNumber, const char *sourceFile,
    int sourceLine) {
  return BeginExternalFormattedIO<Direction::Output>(
      format, formatLength, unitNumber, sourceFile, sourceLine);
}

Cookie IONAME(BeginExternalFormattedInput)(const char *format,
    std::size_t formatLength, ExternalUnit unitNumber, const char *sourceFile,
    int sourceLine) {
  return BeginExternalFormattedIO<Direction::Input>(
      format, formatLength, unitNumber, sourceFile, sourceLine);
}

template <Direction DIR>
Cookie BeginUnformattedIO(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{ExternalFileUnit::LookUpOrCreateAnonymous(
      unitNumber, DIR, true /*unformatted*/, terminator)};
  if (!unit.isUnformatted) {
    terminator.Crash("Unformatted output attempted on formatted file");
  }
  IoStatementState &io{unit.BeginIoStatement<UnformattedIoStatementState<DIR>>(
      unit, sourceFile, sourceLine)};
  IoErrorHandler handler{terminator};
  unit.SetDirection(DIR, handler);
  if constexpr (DIR == Direction::Input) {
    unit.BeginReadingRecord(handler);
  } else {
    if (unit.access == Access::Sequential && !unit.isFixedRecordLength) {
      // Create space for (sub)record header to be completed by
      // UnformattedIoStatementState<Direction::Output>::EndIoStatement()
      io.Emit("\0\0\0\0", 4); // placeholder for record length header
    }
  }
  return &io;
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
  bool wasExtant{false};
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{
      ExternalFileUnit::LookUpOrCreate(unitNumber, terminator, wasExtant)};
  return &unit.BeginIoStatement<OpenStatementState>(
      unit, wasExtant, sourceFile, sourceLine);
}

Cookie IONAME(BeginOpenNewUnit)( // OPEN(NEWUNIT=j)
    const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  bool ignored{false};
  ExternalFileUnit &unit{ExternalFileUnit::LookUpOrCreate(
      ExternalFileUnit::NewUnit(terminator), terminator, ignored)};
  return &unit.BeginIoStatement<OpenStatementState>(
      unit, false /*was an existing file*/, sourceFile, sourceLine);
}

Cookie IONAME(BeginClose)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUpForClose(unitNumber)}) {
    return &unit->BeginIoStatement<CloseStatementState>(
        *unit, sourceFile, sourceLine);
  } else {
    // CLOSE(UNIT=bad unit) is just a no-op
    Terminator oom{sourceFile, sourceLine};
    return &New<NoopCloseStatementState>{oom}(sourceFile, sourceLine)
                .release()
                ->ioStatementState();
  }
}

Cookie IONAME(BeginFlush)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{
      ExternalFileUnit::LookUpOrCrash(unitNumber, terminator)};
  return &unit.BeginIoStatement<ExternalMiscIoStatementState>(
      unit, ExternalMiscIoStatementState::Flush, sourceFile, sourceLine);
}

Cookie IONAME(BeginBackspace)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{
      ExternalFileUnit::LookUpOrCrash(unitNumber, terminator)};
  return &unit.BeginIoStatement<ExternalMiscIoStatementState>(
      unit, ExternalMiscIoStatementState::Backspace, sourceFile, sourceLine);
}

Cookie IONAME(BeginEndfile)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{
      ExternalFileUnit::LookUpOrCrash(unitNumber, terminator)};
  return &unit.BeginIoStatement<ExternalMiscIoStatementState>(
      unit, ExternalMiscIoStatementState::Endfile, sourceFile, sourceLine);
}

Cookie IONAME(BeginRewind)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{
      ExternalFileUnit::LookUpOrCrash(unitNumber, terminator)};
  return &unit.BeginIoStatement<ExternalMiscIoStatementState>(
      unit, ExternalMiscIoStatementState::Rewind, sourceFile, sourceLine);
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
  ConnectionState &connection{io.GetConnectionState()};
  connection.nonAdvancing =
      !YesOrNo(keyword, length, "ADVANCE", io.GetIoErrorHandler());
  if (connection.nonAdvancing && connection.access == Access::Direct) {
    io.GetIoErrorHandler().SignalError(
        "Non-advancing I/O attempted on direct access file");
  }
  return true;
}

bool IONAME(SetBlank)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  static const char *keywords[]{"NULL", "ZERO", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    connection.modes.editingFlags &= ~blankZero;
    return true;
  case 1:
    connection.modes.editingFlags |= blankZero;
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
  ConnectionState &connection{io.GetConnectionState()};
  static const char *keywords[]{"COMMA", "POINT", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    connection.modes.editingFlags |= decimalComma;
    return true;
  case 1:
    connection.modes.editingFlags &= ~decimalComma;
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid DECIMAL='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetDelim)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  static const char *keywords[]{"APOSTROPHE", "QUOTE", "NONE", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    connection.modes.delim = '\'';
    return true;
  case 1:
    connection.modes.delim = '"';
    return true;
  case 2:
    connection.modes.delim = '\0';
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid DELIM='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetPad)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  connection.modes.pad =
      YesOrNo(keyword, length, "PAD", io.GetIoErrorHandler());
  return true;
}

bool IONAME(SetPos)(Cookie cookie, std::int64_t pos) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  if (connection.access != Access::Stream) {
    io.GetIoErrorHandler().SignalError(
        "REC= may not appear unless ACCESS='STREAM'");
    return false;
  }
  if (pos < 1) {
    io.GetIoErrorHandler().SignalError(
        "POS=%zd is invalid", static_cast<std::intmax_t>(pos));
    return false;
  }
  if (auto *unit{io.GetExternalFileUnit()}) {
    unit->SetPosition(pos);
    return true;
  }
  io.GetIoErrorHandler().Crash("SetPos() on internal unit");
  return false;
}

bool IONAME(SetRec)(Cookie cookie, std::int64_t rec) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  if (connection.access != Access::Direct) {
    io.GetIoErrorHandler().SignalError(
        "REC= may not appear unless ACCESS='DIRECT'");
    return false;
  }
  if (!connection.isFixedRecordLength || !connection.recordLength) {
    io.GetIoErrorHandler().SignalError("RECL= was not specified");
    return false;
  }
  if (rec < 1) {
    io.GetIoErrorHandler().SignalError(
        "REC=%zd is invalid", static_cast<std::intmax_t>(rec));
    return false;
  }
  connection.currentRecordNumber = rec;
  if (auto *unit{io.GetExternalFileUnit()}) {
    unit->SetPosition(rec * *connection.recordLength);
  }
  return true;
}

bool IONAME(SetRound)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  static const char *keywords[]{"UP", "DOWN", "ZERO", "NEAREST", "COMPATIBLE",
      "PROCESSOR_DEFINED", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    connection.modes.round = decimal::RoundUp;
    return true;
  case 1:
    connection.modes.round = decimal::RoundDown;
    return true;
  case 2:
    connection.modes.round = decimal::RoundToZero;
    return true;
  case 3:
    connection.modes.round = decimal::RoundNearest;
    return true;
  case 4:
    connection.modes.round = decimal::RoundCompatible;
    return true;
  case 5:
    connection.modes.round = executionEnvironment.defaultOutputRoundingMode;
    return true;
  default:
    io.GetIoErrorHandler().SignalError(IostatErrorInKeyword,
        "Invalid ROUND='%.*s'", static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetSign)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  ConnectionState &connection{io.GetConnectionState()};
  static const char *keywords[]{"PLUS", "YES", "PROCESSOR_DEFINED", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    connection.modes.editingFlags |= signPlus;
    return true;
  case 1:
  case 2: // processor default is SS
    connection.modes.editingFlags &= ~signPlus;
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
    io.GetIoErrorHandler().Crash(
        "SetAccess() called when not in an OPEN statement");
  }
  ConnectionState &connection{open->GetConnectionState()};
  Access access{connection.access};
  static const char *keywords[]{"SEQUENTIAL", "DIRECT", "STREAM", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    access = Access::Sequential;
    break;
  case 1:
    access = Access::Direct;
    break;
  case 2:
    access = Access::Stream;
    break;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid ACCESS='%.*s'",
        static_cast<int>(length), keyword);
  }
  if (access != connection.access) {
    if (open->wasExtant()) {
      open->SignalError("ACCESS= may not be changed on an open unit");
    }
    connection.access = access;
  }
  return true;
}

bool IONAME(SetAction)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    io.GetIoErrorHandler().Crash(
        "SetAction() called when not in an OPEN statement");
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
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    io.GetIoErrorHandler().Crash(
        "SetAsynchronous() called when not in an OPEN statement");
  }
  static const char *keywords[]{"YES", "NO", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    open->unit().set_mayAsynchronous(true);
    return true;
  case 1:
    open->unit().set_mayAsynchronous(false);
    return true;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid ASYNCHRONOUS='%.*s'",
        static_cast<int>(length), keyword);
    return false;
  }
}

bool IONAME(SetConvert)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    io.GetIoErrorHandler().Crash(
        "SetConvert() called when not in an OPEN statement");
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
    io.GetIoErrorHandler().Crash(
        "SetEncoding() called when not in an OPEN statement");
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
    io.GetIoErrorHandler().Crash(
        "SetEncoding() called when not in an OPEN statement");
  }
  bool isUnformatted{false};
  static const char *keywords[]{"FORMATTED", "UNFORMATTED", nullptr};
  switch (IdentifyValue(keyword, length, keywords)) {
  case 0:
    isUnformatted = false;
    break;
  case 1:
    isUnformatted = true;
    break;
  default:
    open->SignalError(IostatErrorInKeyword, "Invalid FORM='%.*s'",
        static_cast<int>(length), keyword);
  }
  if (isUnformatted != open->unit().isUnformatted) {
    if (open->wasExtant()) {
      open->SignalError("FORM= may not be changed on an open unit");
    }
    open->unit().isUnformatted = isUnformatted;
  }
  return true;
}

bool IONAME(SetPosition)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    io.GetIoErrorHandler().Crash(
        "SetPosition() called when not in an OPEN statement");
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
    io.GetIoErrorHandler().Crash(
        "SetRecl() called when not in an OPEN statement");
  }
  if (n <= 0) {
    io.GetIoErrorHandler().SignalError("RECL= must be greater than zero");
  }
  if (open->wasExtant() && open->unit().isFixedRecordLength &&
      open->unit().recordLength.value_or(n) != static_cast<std::int64_t>(n)) {
    open->SignalError("RECL= may not be changed for an open unit");
  }
  open->unit().isFixedRecordLength = true;
  open->unit().recordLength = n;
  return true;
}

bool IONAME(SetStatus)(Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  if (auto *open{io.get_if<OpenStatementState>()}) {
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
  if (io.get_if<NoopCloseStatementState>()) {
    return true; // don't bother validating STATUS= in a no-op CLOSE
  }
  io.GetIoErrorHandler().Crash(
      "SetStatus() called when not in an OPEN or CLOSE statement");
}

bool IONAME(SetFile)(
    Cookie cookie, const char *path, std::size_t chars, int kind) {
  IoStatementState &io{*cookie};
  if (auto *open{io.get_if<OpenStatementState>()}) {
    open->set_path(path, chars, kind);
    return true;
  }
  io.GetIoErrorHandler().Crash(
      "SetFile() called when not in an OPEN statement");
  return false;
}

static bool SetInteger(int &x, int kind, int value) {
  switch (kind) {
  case 1:
    reinterpret_cast<std::int8_t &>(x) = value;
    return true;
  case 2:
    reinterpret_cast<std::int16_t &>(x) = value;
    return true;
  case 4:
    x = value;
    return true;
  case 8:
    reinterpret_cast<std::int64_t &>(x) = value;
    return true;
  default:
    return false;
  }
}

bool IONAME(GetNewUnit)(Cookie cookie, int &unit, int kind) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    io.GetIoErrorHandler().Crash(
        "GetNewUnit() called when not in an OPEN statement");
  }
  if (!SetInteger(unit, kind, open->unit().unitNumber())) {
    open->SignalError("GetNewUnit(): Bad INTEGER kind(%d) for result");
  }
  return true;
}

// Data transfers

bool IONAME(OutputDescriptor)(Cookie cookie, const Descriptor &) {
  IoStatementState &io{*cookie};
  io.GetIoErrorHandler().Crash("OutputDescriptor: not yet implemented"); // TODO
}

bool IONAME(InputDescriptor)(Cookie cookie, const Descriptor &) {
  IoStatementState &io{*cookie};
  io.GetIoErrorHandler().Crash("InputDescriptor: not yet implemented"); // TODO
}

bool IONAME(OutputUnformattedBlock)(Cookie cookie, const char *x,
    std::size_t length, std::size_t elementBytes) {
  IoStatementState &io{*cookie};
  if (auto *unf{io.get_if<UnformattedIoStatementState<Direction::Output>>()}) {
    return unf->Emit(x, length, elementBytes);
  }
  io.GetIoErrorHandler().Crash("OutputUnformattedBlock() called for an I/O "
                               "statement that is not unformatted output");
  return false;
}

bool IONAME(InputUnformattedBlock)(
    Cookie cookie, char *x, std::size_t length, std::size_t elementBytes) {
  IoStatementState &io{*cookie};
  if (auto *unf{io.get_if<UnformattedIoStatementState<Direction::Input>>()}) {
    return unf->Receive(x, length, elementBytes);
  }
  io.GetIoErrorHandler().Crash("InputUnformattedBlock() called for an I/O "
                               "statement that is not unformatted output");
  return false;
}

bool IONAME(OutputInteger64)(Cookie cookie, std::int64_t n) {
  IoStatementState &io{*cookie};
  if (!io.get_if<OutputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "OutputInteger64() called for a non-output I/O statement");
    return false;
  }
  if (auto edit{io.GetNextDataEdit()}) {
    return EditIntegerOutput(io, *edit, n);
  }
  return false;
}

bool IONAME(InputInteger)(Cookie cookie, std::int64_t &n, int kind) {
  IoStatementState &io{*cookie};
  if (!io.get_if<InputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "InputInteger64() called for a non-input I/O statement");
    return false;
  }
  if (auto edit{io.GetNextDataEdit()}) {
    if (edit->descriptor == DataEdit::ListDirectedNullValue) {
      return true;
    }
    return EditIntegerInput(io, *edit, reinterpret_cast<void *>(&n), kind);
  }
  return false;
}

template <int PREC, typename REAL>
static bool OutputReal(Cookie cookie, REAL x) {
  IoStatementState &io{*cookie};
  if (!io.get_if<OutputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "OutputReal() called for a non-output I/O statement");
    return false;
  }
  if (auto edit{io.GetNextDataEdit()}) {
    return RealOutputEditing<PREC>{io, x}.Edit(*edit);
  }
  return false;
}

bool IONAME(OutputReal32)(Cookie cookie, float x) {
  return OutputReal<24, float>(cookie, x);
}

bool IONAME(OutputReal64)(Cookie cookie, double x) {
  return OutputReal<53, double>(cookie, x);
}

template <int PREC, typename REAL>
static bool InputReal(Cookie cookie, REAL &x) {
  IoStatementState &io{*cookie};
  if (!io.get_if<InputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "InputReal() called for a non-input I/O statement");
    return false;
  }
  if (auto edit{io.GetNextDataEdit()}) {
    if (edit->descriptor == DataEdit::ListDirectedNullValue) {
      return true;
    }
    return EditRealInput<PREC>(io, *edit, reinterpret_cast<void *>(&x));
  }
  return false;
}

bool IONAME(InputReal32)(Cookie cookie, float &x) {
  return InputReal<24, float>(cookie, x);
}

bool IONAME(InputReal64)(Cookie cookie, double &x) {
  return InputReal<53, double>(cookie, x);
}

template <int PREC, typename REAL>
static bool OutputComplex(Cookie cookie, REAL r, REAL z) {
  IoStatementState &io{*cookie};
  if (io.get_if<ListDirectedStatementState<Direction::Output>>()) {
    DataEdit real, imaginary;
    real.descriptor = DataEdit::ListDirectedRealPart;
    imaginary.descriptor = DataEdit::ListDirectedImaginaryPart;
    return RealOutputEditing<PREC>{io, r}.Edit(real) &&
        RealOutputEditing<PREC>{io, z}.Edit(imaginary);
  }
  return OutputReal<PREC, REAL>(cookie, r) && OutputReal<PREC, REAL>(cookie, z);
}

bool IONAME(OutputComplex32)(Cookie cookie, float r, float z) {
  return OutputComplex<24, float>(cookie, r, z);
}

bool IONAME(OutputComplex64)(Cookie cookie, double r, double z) {
  return OutputComplex<53, double>(cookie, r, z);
}

template <int PREC, typename REAL>
static bool InputComplex(Cookie cookie, REAL x[2]) {
  IoStatementState &io{*cookie};
  if (!io.get_if<InputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "InputComplex() called for a non-input I/O statement");
    return false;
  }
  for (int j{0}; j < 2; ++j) {
    if (auto edit{io.GetNextDataEdit()}) {
      if (edit->descriptor == DataEdit::ListDirectedNullValue) {
        return true;
      }
      if (!EditRealInput<PREC>(io, *edit, reinterpret_cast<void *>(&x[j]))) {
        return false;
      }
    }
  }
  return true;
}

bool IONAME(InputComplex32)(Cookie cookie, float x[2]) {
  return InputComplex<24, float>(cookie, x);
}

bool IONAME(InputComplex64)(Cookie cookie, double x[2]) {
  return InputComplex<53, double>(cookie, x);
}

bool IONAME(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  IoStatementState &io{*cookie};
  if (!io.get_if<OutputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "OutputAscii() called for a non-output I/O statement");
    return false;
  }
  if (auto *list{io.get_if<ListDirectedStatementState<Direction::Output>>()}) {
    return ListDirectedDefaultCharacterOutput(io, *list, x, length);
  } else if (auto edit{io.GetNextDataEdit()}) {
    return EditDefaultCharacterOutput(io, *edit, x, length);
  } else {
    return false;
  }
}

bool IONAME(InputAscii)(Cookie cookie, char *x, std::size_t length) {
  IoStatementState &io{*cookie};
  if (!io.get_if<InputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "InputAscii() called for a non-input I/O statement");
    return false;
  }
  if (auto edit{io.GetNextDataEdit()}) {
    if (edit->descriptor == DataEdit::ListDirectedNullValue) {
      return true;
    }
    return EditDefaultCharacterInput(io, *edit, x, length);
  }
  return false;
}

bool IONAME(OutputLogical)(Cookie cookie, bool truth) {
  IoStatementState &io{*cookie};
  if (!io.get_if<OutputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "OutputLogical() called for a non-output I/O statement");
    return false;
  }
  if (auto *list{io.get_if<ListDirectedStatementState<Direction::Output>>()}) {
    return ListDirectedLogicalOutput(io, *list, truth);
  } else if (auto edit{io.GetNextDataEdit()}) {
    return EditLogicalOutput(io, *edit, truth);
  } else {
    return false;
  }
}

bool IONAME(InputLogical)(Cookie cookie, bool &truth) {
  IoStatementState &io{*cookie};
  if (!io.get_if<InputStatementState>()) {
    io.GetIoErrorHandler().Crash(
        "InputLogical() called for a non-input I/O statement");
    return false;
  }
  if (auto edit{io.GetNextDataEdit()}) {
    if (edit->descriptor == DataEdit::ListDirectedNullValue) {
      return true;
    }
    return EditLogicalInput(io, *edit, truth);
  }
  return false;
}

void IONAME(GetIoMsg)(Cookie cookie, char *msg, std::size_t length) {
  IoErrorHandler &handler{cookie->GetIoErrorHandler()};
  if (handler.GetIoStat()) { // leave "msg" alone when no error
    handler.GetIoMsg(msg, length);
  }
}

enum Iostat IONAME(EndIoStatement)(Cookie cookie) {
  IoStatementState &io{*cookie};
  return static_cast<enum Iostat>(io.EndIoStatement());
}
} // namespace Fortran::runtime::io
