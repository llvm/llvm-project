//===-- runtime/io-api.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the I/O statement API

#include "io-api.h"
#include "descriptor-io.h"
#include "descriptor.h"
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
  if constexpr (DIR == Direction::Output) {
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
  ExternalFileUnit &unit{ExternalFileUnit::LookUpOrCreateAnonymous(
      unitNumber, Direction::Output, true /*formatted*/, terminator)};
  return &unit.BeginIoStatement<ExternalMiscIoStatementState>(
      unit, ExternalMiscIoStatementState::Endfile, sourceFile, sourceLine);
}

Cookie IONAME(BeginRewind)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  ExternalFileUnit &unit{ExternalFileUnit::LookUpOrCreateAnonymous(
      unitNumber, Direction::Input, true /*formatted*/, terminator)};
  return &unit.BeginIoStatement<ExternalMiscIoStatementState>(
      unit, ExternalMiscIoStatementState::Rewind, sourceFile, sourceLine);
}

Cookie IONAME(BeginInquireUnit)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(unitNumber)}) {
    return &unit->BeginIoStatement<InquireUnitState>(
        *unit, sourceFile, sourceLine);
  } else {
    // INQUIRE(UNIT=unrecognized unit)
    Terminator oom{sourceFile, sourceLine};
    return &New<InquireNoUnitState>{oom}(sourceFile, sourceLine)
                .release()
                ->ioStatementState();
  }
}

Cookie IONAME(BeginInquireFile)(const char *path, std::size_t pathLength,
    const char *sourceFile, int sourceLine) {
  Terminator oom{sourceFile, sourceLine};
  auto trimmed{
      SaveDefaultCharacter(path, TrimTrailingSpaces(path, pathLength), oom)};
  if (ExternalFileUnit * unit{ExternalFileUnit::LookUp(trimmed.get())}) {
    // INQUIRE(FILE=) to a connected unit
    return &unit->BeginIoStatement<InquireUnitState>(
        *unit, sourceFile, sourceLine);
  } else {
    return &New<InquireUnconnectedFileState>{oom}(
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
  static const char *keywords[]{"SEQUENTIAL", "DIRECT", "STREAM", nullptr};
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

bool IONAME(SetCarriagecontrol)(
    Cookie cookie, const char *keyword, std::size_t length) {
  IoStatementState &io{*cookie};
  auto *open{io.get_if<OpenStatementState>()};
  if (!open) {
    io.GetIoErrorHandler().Crash(
        "SetCarriageControl() called when not in an OPEN statement");
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
        "SetForm() called when not in an OPEN statement");
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

bool IONAME(SetFile)(Cookie cookie, const char *path, std::size_t chars) {
  IoStatementState &io{*cookie};
  if (auto *open{io.get_if<OpenStatementState>()}) {
    open->set_path(path, chars);
    return true;
  }
  io.GetIoErrorHandler().Crash(
      "SetFile() called when not in an OPEN statement");
  return false;
}

template <typename INT>
static bool SetInteger(INT &x, int kind, std::int64_t value) {
  switch (kind) {
  case 1:
    reinterpret_cast<std::int8_t &>(x) = value;
    return true;
  case 2:
    reinterpret_cast<std::int16_t &>(x) = value;
    return true;
  case 4:
    reinterpret_cast<std::int32_t &>(x) = value;
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

bool IONAME(OutputDescriptor)(Cookie cookie, const Descriptor &descriptor) {
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputDescriptor)(Cookie cookie, const Descriptor &descriptor) {
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
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
  io.BeginReadingRecord();
  if (auto *unf{io.get_if<UnformattedIoStatementState<Direction::Input>>()}) {
    return unf->Receive(x, length, elementBytes);
  }
  io.GetIoErrorHandler().Crash("InputUnformattedBlock() called for an I/O "
                               "statement that is not unformatted output");
  return false;
}

bool IONAME(OutputInteger64)(Cookie cookie, std::int64_t n) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, 8, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputInteger)(Cookie cookie, std::int64_t &n, int kind) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Integer, kind, reinterpret_cast<void *>(&n), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputReal32)(Cookie cookie, float x) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 4, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputReal64)(Cookie cookie, double x) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 8, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputReal32)(Cookie cookie, float &x) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 4, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(InputReal64)(Cookie cookie, double &x) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(TypeCategory::Real, 8, reinterpret_cast<void *>(&x), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputComplex32)(Cookie cookie, float r, float i) {
  float z[2]{r, i};
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 4, reinterpret_cast<void *>(&z), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(OutputComplex64)(Cookie cookie, double r, double i) {
  double z[2]{r, i};
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 8, reinterpret_cast<void *>(&z), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputComplex32)(Cookie cookie, float z[2]) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 4, reinterpret_cast<void *>(z), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(InputComplex64)(Cookie cookie, double z[2]) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Complex, 8, reinterpret_cast<void *>(z), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      1, length, reinterpret_cast<void *>(const_cast<char *>(x)), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputAscii)(Cookie cookie, char *x, std::size_t length) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(1, length, reinterpret_cast<void *>(x), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

bool IONAME(OutputLogical)(Cookie cookie, bool truth) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Logical, 1, reinterpret_cast<void *>(&truth), 0);
  return descr::DescriptorIO<Direction::Output>(*cookie, descriptor);
}

bool IONAME(InputLogical)(Cookie cookie, bool &truth) {
  StaticDescriptor staticDescriptor;
  Descriptor &descriptor{staticDescriptor.descriptor()};
  descriptor.Establish(
      TypeCategory::Logical, 1, reinterpret_cast<void *>(&truth), 0);
  return descr::DescriptorIO<Direction::Input>(*cookie, descriptor);
}

void IONAME(GetIoMsg)(Cookie cookie, char *msg, std::size_t length) {
  IoErrorHandler &handler{cookie->GetIoErrorHandler()};
  if (handler.GetIoStat()) { // leave "msg" alone when no error
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
    SetInteger(result, kind, n);
    return true;
  }
  return false;
}

enum Iostat IONAME(EndIoStatement)(Cookie cookie) {
  IoStatementState &io{*cookie};
  return static_cast<enum Iostat>(io.EndIoStatement());
}
} // namespace Fortran::runtime::io
