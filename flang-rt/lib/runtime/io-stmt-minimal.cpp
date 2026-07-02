//===-- lib/runtime/io-stmt-minimal.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the subset of the I/O statement API needed for basic
// list-directed output (PRINT *) of intrinsic types.

#include "unit.h"
#include "flang-rt/runtime/io-stmt.h"
#include <algorithm>

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

ExternalIoStatementBase::ExternalIoStatementBase(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{unit} {}

template <Direction DIR>
ExternalIoStatementState<DIR>::ExternalIoStatementState(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
    : ExternalIoStatementBase{unit, sourceFile, sourceLine},
      mutableModes_{unit.modes} {
  if constexpr (DIR == Direction::Output) {
    if (!unit.isUnformatted.has_value()) {
      unit.isUnformatted = false;
    }
    if (!unit.openRecl.has_value()) {
      unit.openRecl = 79;
    }
    unit.SetDirection(Direction::Output);
    unit.furthestPositionInRecord =
        std::max(unit.furthestPositionInRecord, unit.positionInRecord);
  }
}

template ExternalIoStatementState<Direction::Output>::ExternalIoStatementState(
    ExternalFileUnit &, const char *, int);

common::optional<DataEdit> IoStatementState::GetNextDataEdit(int maxRepeat) {
  if (auto *listOutput{
          get_if<ExternalListIoStatementState<Direction::Output>>()}) {
    return listOutput->GetNextDataEdit(*this, maxRepeat);
  }
  return common::nullopt;
}

common::optional<DataEdit>
ListDirectedStatementState<Direction::Output>::GetNextDataEdit(
    IoStatementState &io, int maxRepeat) {
  DataEdit edit;
  edit.descriptor = DataEdit::ListDirected;
  edit.repeat = maxRepeat;
  edit.modes = io.mutableModes();
  return edit;
}

MutableModes &IoStatementState::mutableModes() {
  if (auto *listOutput{
          get_if<ExternalListIoStatementState<Direction::Output>>()}) {
    return listOutput->mutableModes();
  }
  return GetConnectionState().modes;
}

ConnectionState &IoStatementState::GetConnectionState() {
  return get_if<ExternalListIoStatementState<Direction::Output>>()
      ->GetConnectionState();
}

ConnectionState &ExternalIoStatementBase::GetConnectionState() { return unit_; }

template <Direction DIR>
bool ExternalIoStatementState<DIR>::Emit(
    const char *data, std::size_t bytes, std::size_t elementBytes) {
  return unit().Emit(data, bytes, elementBytes, *this);
}

template bool ExternalIoStatementState<Direction::Output>::Emit(
    const char *, std::size_t, std::size_t);

bool IoStatementState::Emit(
    const char *data, std::size_t bytes, std::size_t elementBytes) {
  return get_if<ExternalListIoStatementState<Direction::Output>>()->Emit(
      data, bytes, elementBytes);
}

IoErrorHandler &IoStatementState::GetIoErrorHandler() const {
  return *get_if<ExternalListIoStatementState<Direction::Output>>();
}

template <Direction DIR>
void ExternalIoStatementState<DIR>::CompleteOperation() {
  if (completedOperation()) {
    return;
  }
  if constexpr (DIR == Direction::Output) {
    if (mutableModes().nonAdvancing) {
      if (unit().positionInRecord > unit().furthestPositionInRecord) {
        unit().Emit("", 0, 1, *this);
      }
      unit().leftTabLimit = unit().positionInRecord;
    } else {
      unit().AdvanceRecord(*this);
    }
    unit().FlushIfTerminal(*this);
  }
  IoStatementBase::CompleteOperation();
}

template void ExternalIoStatementState<Direction::Output>::CompleteOperation();

template <Direction DIR> int ExternalIoStatementState<DIR>::EndIoStatement() {
  CompleteOperation();
  return ExternalIoStatementBase::EndIoStatement();
}

template int ExternalIoStatementState<Direction::Output>::EndIoStatement();

template <Direction DIR>
int ExternalListIoStatementState<DIR>::EndIoStatement() {
  return ExternalIoStatementState<DIR>::EndIoStatement();
}

template int ExternalListIoStatementState<Direction::Output>::EndIoStatement();

int IoStatementState::EndIoStatement() {
  return get_if<ExternalListIoStatementState<Direction::Output>>()
      ->EndIoStatement();
}

int ExternalIoStatementBase::EndIoStatement() {
  CompleteOperation();
  auto result{IoStatementBase::EndIoStatement()};
#if !defined(RT_USE_PSEUDO_FILE_UNIT)
  unit_.EndIoStatement();
#else
  unit_.~ExternalFileUnit();
  FreeMemory(&unit_);
#endif
  return result;
}

void IoErrorHandler::SignalEnd() {}
void IoErrorHandler::SignalEor() {}
void IoErrorHandler::SignalError(int) {}
void IoErrorHandler::SignalError(int, const char *, ...) {}

bool IoStatementState::AdvanceRecord(int n) {
  return get_if<ExternalListIoStatementState<Direction::Output>>()
      ->AdvanceRecord(n);
}

template <Direction DIR>
bool ExternalIoStatementState<DIR>::AdvanceRecord(int n) {
  while (n-- > 0) {
    if (!unit().AdvanceRecord(*this)) {
      return false;
    }
  }
  return true;
}

template bool ExternalIoStatementState<Direction::Output>::AdvanceRecord(int);

bool ListDirectedStatementState<Direction::Output>::EmitLeadingSpaceOrAdvance(
    IoStatementState &io, std::size_t length, bool isCharacter) {
  const ConnectionState &connection{io.GetConnectionState()};
  int space{connection.positionInRecord == 0 ||
      !(isCharacter && lastWasUndelimitedCharacter())};
  set_lastWasUndelimitedCharacter(false);
  if (connection.NeedAdvance(space + length)) {
    return io.AdvanceRecord();
  }
  if (space) {
    return io.Emit(" ", 1);
  }
  return true;
}

void IoStatementState::BackspaceRecord() {}

template <Direction DIR> void ExternalIoStatementState<DIR>::BackspaceRecord() {
  unit().BackspaceRecord(*this);
}

template void ExternalIoStatementState<Direction::Output>::BackspaceRecord();

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
