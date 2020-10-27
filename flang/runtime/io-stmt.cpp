//===-- runtime/io-stmt.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-stmt.h"
#include "connection.h"
#include "format.h"
#include "memory.h"
#include "tools.h"
#include "unit.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace Fortran::runtime::io {

int IoStatementBase::EndIoStatement() { return GetIoStat(); }

std::optional<DataEdit> IoStatementBase::GetNextDataEdit(
    IoStatementState &, int) {
  return std::nullopt;
}

template <Direction DIR, typename CHAR>
InternalIoStatementState<DIR, CHAR>::InternalIoStatementState(
    Buffer scalar, std::size_t length, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{scalar, length} {}

template <Direction DIR, typename CHAR>
InternalIoStatementState<DIR, CHAR>::InternalIoStatementState(
    const Descriptor &d, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{d, *this} {}

template <Direction DIR, typename CHAR>
bool InternalIoStatementState<DIR, CHAR>::Emit(
    const CharType *data, std::size_t chars, std::size_t /*elementBytes*/) {
  if constexpr (DIR == Direction::Input) {
    Crash("InternalIoStatementState<Direction::Input>::Emit() called");
    return false;
  }
  return unit_.Emit(data, chars, *this);
}

template <Direction DIR, typename CHAR>
std::optional<char32_t> InternalIoStatementState<DIR, CHAR>::GetCurrentChar() {
  if constexpr (DIR == Direction::Output) {
    Crash(
        "InternalIoStatementState<Direction::Output>::GetCurrentChar() called");
    return std::nullopt;
  }
  return unit_.GetCurrentChar(*this);
}

template <Direction DIR, typename CHAR>
bool InternalIoStatementState<DIR, CHAR>::AdvanceRecord(int n) {
  while (n-- > 0) {
    if (!unit_.AdvanceRecord(*this)) {
      return false;
    }
  }
  return true;
}

template <Direction DIR, typename CHAR>
void InternalIoStatementState<DIR, CHAR>::BackspaceRecord() {
  unit_.BackspaceRecord(*this);
}

template <Direction DIR, typename CHAR>
int InternalIoStatementState<DIR, CHAR>::EndIoStatement() {
  if constexpr (DIR == Direction::Output) {
    unit_.EndIoStatement(); // fill
  }
  auto result{IoStatementBase::EndIoStatement()};
  if (free_) {
    FreeMemory(this);
  }
  return result;
}

template <Direction DIR, typename CHAR>
void InternalIoStatementState<DIR, CHAR>::HandleAbsolutePosition(
    std::int64_t n) {
  return unit_.HandleAbsolutePosition(n);
}

template <Direction DIR, typename CHAR>
void InternalIoStatementState<DIR, CHAR>::HandleRelativePosition(
    std::int64_t n) {
  return unit_.HandleRelativePosition(n);
}

template <Direction DIR, typename CHAR>
InternalFormattedIoStatementState<DIR, CHAR>::InternalFormattedIoStatementState(
    Buffer buffer, std::size_t length, const CHAR *format,
    std::size_t formatLength, const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR, CHAR>{buffer, length, sourceFile,
          sourceLine},
      ioStatementState_{*this}, format_{*this, format, formatLength} {}

template <Direction DIR, typename CHAR>
InternalFormattedIoStatementState<DIR, CHAR>::InternalFormattedIoStatementState(
    const Descriptor &d, const CHAR *format, std::size_t formatLength,
    const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR, CHAR>{d, sourceFile, sourceLine},
      ioStatementState_{*this}, format_{*this, format, formatLength} {}

template <Direction DIR, typename CHAR>
int InternalFormattedIoStatementState<DIR, CHAR>::EndIoStatement() {
  if constexpr (DIR == Direction::Output) {
    format_.Finish(*this); // ignore any remaining input positioning actions
  }
  return InternalIoStatementState<DIR, CHAR>::EndIoStatement();
}

template <Direction DIR, typename CHAR>
InternalListIoStatementState<DIR, CHAR>::InternalListIoStatementState(
    Buffer buffer, std::size_t length, const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR, CharType>{buffer, length, sourceFile,
          sourceLine},
      ioStatementState_{*this} {}

template <Direction DIR, typename CHAR>
InternalListIoStatementState<DIR, CHAR>::InternalListIoStatementState(
    const Descriptor &d, const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR, CharType>{d, sourceFile, sourceLine},
      ioStatementState_{*this} {}

ExternalIoStatementBase::ExternalIoStatementBase(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{unit} {}

MutableModes &ExternalIoStatementBase::mutableModes() { return unit_.modes; }

ConnectionState &ExternalIoStatementBase::GetConnectionState() { return unit_; }

int ExternalIoStatementBase::EndIoStatement() {
  if (unit_.nonAdvancing) {
    unit_.leftTabLimit = unit_.furthestPositionInRecord;
    unit_.nonAdvancing = false;
  } else {
    unit_.leftTabLimit.reset();
  }
  auto result{IoStatementBase::EndIoStatement()};
  unit_.EndIoStatement(); // annihilates *this in unit_.u_
  return result;
}

void OpenStatementState::set_path(
    const char *path, std::size_t length, int kind) {
  if (kind != 1) { // TODO
    Crash("OPEN: FILE= with unimplemented: CHARACTER(KIND=%d)", kind);
  }
  std::size_t bytes{length * kind}; // TODO: UTF-8 encoding of Unicode path
  path_ = SaveDefaultCharacter(path, bytes, *this);
  pathLength_ = length;
}

int OpenStatementState::EndIoStatement() {
  if (wasExtant_ && status_ && *status_ != OpenStatus::Old) {
    SignalError("OPEN statement for connected unit may not have STATUS= other "
                "than 'OLD'");
  }
  unit().OpenUnit(status_.value_or(OpenStatus::Unknown), action_, position_,
      std::move(path_), pathLength_, convert_, *this);
  return ExternalIoStatementBase::EndIoStatement();
}

int CloseStatementState::EndIoStatement() {
  int result{ExternalIoStatementBase::EndIoStatement()};
  unit().CloseUnit(status_, *this);
  unit().DestroyClosed();
  return result;
}

int NoopCloseStatementState::EndIoStatement() {
  auto result{IoStatementBase::EndIoStatement()};
  FreeMemory(this);
  return result;
}

template <Direction DIR> int ExternalIoStatementState<DIR>::EndIoStatement() {
  if (!unit().nonAdvancing) {
    unit().AdvanceRecord(*this);
  }
  if constexpr (DIR == Direction::Output) {
    unit().FlushIfTerminal(*this);
  }
  return ExternalIoStatementBase::EndIoStatement();
}

template <Direction DIR>
bool ExternalIoStatementState<DIR>::Emit(
    const char *data, std::size_t bytes, std::size_t elementBytes) {
  if constexpr (DIR == Direction::Input) {
    Crash("ExternalIoStatementState::Emit(char) called for input statement");
  }
  return unit().Emit(data, bytes, elementBytes, *this);
}

template <Direction DIR>
bool ExternalIoStatementState<DIR>::Emit(
    const char16_t *data, std::size_t chars) {
  if constexpr (DIR == Direction::Input) {
    Crash(
        "ExternalIoStatementState::Emit(char16_t) called for input statement");
  }
  // TODO: UTF-8 encoding
  return unit().Emit(reinterpret_cast<const char *>(data), chars * sizeof *data,
      static_cast<int>(sizeof *data), *this);
}

template <Direction DIR>
bool ExternalIoStatementState<DIR>::Emit(
    const char32_t *data, std::size_t chars) {
  if constexpr (DIR == Direction::Input) {
    Crash(
        "ExternalIoStatementState::Emit(char32_t) called for input statement");
  }
  // TODO: UTF-8 encoding
  return unit().Emit(reinterpret_cast<const char *>(data), chars * sizeof *data,
      static_cast<int>(sizeof *data), *this);
}

template <Direction DIR>
std::optional<char32_t> ExternalIoStatementState<DIR>::GetCurrentChar() {
  if constexpr (DIR == Direction::Output) {
    Crash(
        "ExternalIoStatementState<Direction::Output>::GetCurrentChar() called");
  }
  return unit().GetCurrentChar(*this);
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

template <Direction DIR> void ExternalIoStatementState<DIR>::BackspaceRecord() {
  unit().BackspaceRecord(*this);
}

template <Direction DIR>
void ExternalIoStatementState<DIR>::HandleAbsolutePosition(std::int64_t n) {
  return unit().HandleAbsolutePosition(n);
}

template <Direction DIR>
void ExternalIoStatementState<DIR>::HandleRelativePosition(std::int64_t n) {
  return unit().HandleRelativePosition(n);
}

template <Direction DIR, typename CHAR>
ExternalFormattedIoStatementState<DIR, CHAR>::ExternalFormattedIoStatementState(
    ExternalFileUnit &unit, const CHAR *format, std::size_t formatLength,
    const char *sourceFile, int sourceLine)
    : ExternalIoStatementState<DIR>{unit, sourceFile, sourceLine},
      mutableModes_{unit.modes}, format_{*this, format, formatLength} {}

template <Direction DIR, typename CHAR>
int ExternalFormattedIoStatementState<DIR, CHAR>::EndIoStatement() {
  format_.Finish(*this);
  return ExternalIoStatementState<DIR>::EndIoStatement();
}

std::optional<DataEdit> IoStatementState::GetNextDataEdit(int n) {
  return std::visit(
      [&](auto &x) { return x.get().GetNextDataEdit(*this, n); }, u_);
}

bool IoStatementState::Emit(
    const char *data, std::size_t n, std::size_t elementBytes) {
  return std::visit(
      [=](auto &x) { return x.get().Emit(data, n, elementBytes); }, u_);
}

std::optional<char32_t> IoStatementState::GetCurrentChar() {
  return std::visit([&](auto &x) { return x.get().GetCurrentChar(); }, u_);
}

bool IoStatementState::AdvanceRecord(int n) {
  return std::visit([=](auto &x) { return x.get().AdvanceRecord(n); }, u_);
}

void IoStatementState::BackspaceRecord() {
  std::visit([](auto &x) { x.get().BackspaceRecord(); }, u_);
}

void IoStatementState::HandleRelativePosition(std::int64_t n) {
  std::visit([=](auto &x) { x.get().HandleRelativePosition(n); }, u_);
}

int IoStatementState::EndIoStatement() {
  return std::visit([](auto &x) { return x.get().EndIoStatement(); }, u_);
}

ConnectionState &IoStatementState::GetConnectionState() {
  return std::visit(
      [](auto &x) -> ConnectionState & { return x.get().GetConnectionState(); },
      u_);
}

MutableModes &IoStatementState::mutableModes() {
  return std::visit(
      [](auto &x) -> MutableModes & { return x.get().mutableModes(); }, u_);
}

IoErrorHandler &IoStatementState::GetIoErrorHandler() const {
  return std::visit(
      [](auto &x) -> IoErrorHandler & {
        return static_cast<IoErrorHandler &>(x.get());
      },
      u_);
}

ExternalFileUnit *IoStatementState::GetExternalFileUnit() const {
  return std::visit([](auto &x) { return x.get().GetExternalFileUnit(); }, u_);
}

bool IoStatementState::EmitRepeated(char ch, std::size_t n) {
  return std::visit(
      [=](auto &x) {
        for (std::size_t j{0}; j < n; ++j) {
          if (!x.get().Emit(&ch, 1)) {
            return false;
          }
        }
        return true;
      },
      u_);
}

bool IoStatementState::EmitField(
    const char *p, std::size_t length, std::size_t width) {
  if (width <= 0) {
    width = static_cast<int>(length);
  }
  if (length > static_cast<std::size_t>(width)) {
    return EmitRepeated('*', width);
  } else {
    return EmitRepeated(' ', static_cast<int>(width - length)) &&
        Emit(p, length);
  }
}

std::optional<char32_t> IoStatementState::SkipSpaces(
    std::optional<int> &remaining) {
  while (!remaining || *remaining > 0) {
    if (auto ch{GetCurrentChar()}) {
      if (*ch != ' ' && *ch != '\t') {
        return ch;
      }
      HandleRelativePosition(1);
      if (remaining) {
        --*remaining;
      }
    } else {
      break;
    }
  }
  return std::nullopt;
}

std::optional<char32_t> IoStatementState::NextInField(
    std::optional<int> &remaining) {
  if (!remaining) { // list-directed or namelist: check for separators
    if (auto next{GetCurrentChar()}) {
      switch (*next) {
      case ' ':
      case '\t':
      case ',':
      case ';':
      case '/':
      case '(':
      case ')':
      case '\'':
      case '"':
      case '*':
      case '\n': // for stream access
        break;
      default:
        HandleRelativePosition(1);
        return next;
      }
    }
  } else if (*remaining > 0) {
    if (auto next{GetCurrentChar()}) {
      --*remaining;
      HandleRelativePosition(1);
      return next;
    }
    const ConnectionState &connection{GetConnectionState()};
    if (!connection.IsAtEOF() && connection.isFixedRecordLength &&
        connection.recordLength &&
        connection.positionInRecord >= *connection.recordLength) {
      if (connection.modes.pad) { // PAD='YES'
        --*remaining;
        return std::optional<char32_t>{' '};
      }
      IoErrorHandler &handler{GetIoErrorHandler()};
      if (connection.nonAdvancing) {
        handler.SignalEor();
      } else {
        handler.SignalError(IostatRecordReadOverrun);
      }
    }
  }
  return std::nullopt;
}

std::optional<char32_t> IoStatementState::GetNextNonBlank() {
  auto ch{GetCurrentChar()};
  while (!ch || *ch == ' ' || *ch == '\t') {
    if (ch) {
      HandleRelativePosition(1);
    } else if (!AdvanceRecord()) {
      return std::nullopt;
    }
    ch = GetCurrentChar();
  }
  return ch;
}

bool ListDirectedStatementState<Direction::Output>::NeedAdvance(
    const ConnectionState &connection, std::size_t width) const {
  return connection.positionInRecord > 0 &&
      width > connection.RemainingSpaceInRecord();
}

bool ListDirectedStatementState<Direction::Output>::EmitLeadingSpaceOrAdvance(
    IoStatementState &io, std::size_t length, bool isCharacter) {
  if (length == 0) {
    return true;
  }
  const ConnectionState &connection{io.GetConnectionState()};
  int space{connection.positionInRecord == 0 ||
      !(isCharacter && lastWasUndelimitedCharacter)};
  lastWasUndelimitedCharacter = false;
  if (NeedAdvance(connection, space + length)) {
    return io.AdvanceRecord();
  }
  if (space) {
    return io.Emit(" ", 1);
  }
  return true;
}

std::optional<DataEdit>
ListDirectedStatementState<Direction::Output>::GetNextDataEdit(
    IoStatementState &io, int maxRepeat) {
  DataEdit edit;
  edit.descriptor = DataEdit::ListDirected;
  edit.repeat = maxRepeat;
  edit.modes = io.mutableModes();
  return edit;
}

std::optional<DataEdit>
ListDirectedStatementState<Direction::Input>::GetNextDataEdit(
    IoStatementState &io, int maxRepeat) {
  // N.B. list-directed transfers cannot be nonadvancing (C1221)
  ConnectionState &connection{io.GetConnectionState()};
  DataEdit edit;
  edit.descriptor = DataEdit::ListDirected;
  edit.repeat = 1; // may be overridden below
  edit.modes = connection.modes;
  if (hitSlash_) { // everything after '/' is nullified
    edit.descriptor = DataEdit::ListDirectedNullValue;
    return edit;
  }
  char32_t comma{','};
  if (io.mutableModes().editingFlags & decimalComma) {
    comma = ';';
  }
  if (remaining_ > 0 && !realPart_) { // "r*c" repetition in progress
    while (connection.currentRecordNumber > initialRecordNumber_) {
      io.BackspaceRecord();
    }
    connection.HandleAbsolutePosition(initialPositionInRecord_);
    if (!imaginaryPart_) {
      edit.repeat = std::min<int>(remaining_, maxRepeat);
      auto ch{io.GetNextNonBlank()};
      if (!ch || *ch == ' ' || *ch == '\t' || *ch == comma) {
        // "r*" repeated null
        edit.descriptor = DataEdit::ListDirectedNullValue;
      }
    }
    remaining_ -= edit.repeat;
    return edit;
  }
  // Skip separators, handle a "r*c" repeat count; see 13.10.2 in Fortran 2018
  auto ch{io.GetNextNonBlank()};
  if (imaginaryPart_) {
    imaginaryPart_ = false;
    if (ch && *ch == ')') {
      io.HandleRelativePosition(1);
      ch = io.GetNextNonBlank();
    }
  } else if (realPart_) {
    realPart_ = false;
    imaginaryPart_ = true;
    edit.descriptor = DataEdit::ListDirectedImaginaryPart;
  }
  if (!ch) {
    return std::nullopt;
  }
  if (*ch == '/') {
    hitSlash_ = true;
    edit.descriptor = DataEdit::ListDirectedNullValue;
    return edit;
  }
  bool isFirstItem{isFirstItem_};
  isFirstItem_ = false;
  if (*ch == comma) {
    if (isFirstItem) {
      edit.descriptor = DataEdit::ListDirectedNullValue;
      return edit;
    }
    // Consume comma & whitespace after previous item.
    io.HandleRelativePosition(1);
    ch = io.GetNextNonBlank();
    if (!ch) {
      return std::nullopt;
    }
    if (*ch == comma || *ch == '/') {
      edit.descriptor = DataEdit::ListDirectedNullValue;
      return edit;
    }
  }
  if (imaginaryPart_) { // can't repeat components
    return edit;
  }
  if (*ch >= '0' && *ch <= '9') { // look for "r*" repetition count
    auto start{connection.positionInRecord};
    int r{0};
    do {
      static auto constexpr clamp{(std::numeric_limits<int>::max() - '9') / 10};
      if (r >= clamp) {
        r = 0;
        break;
      }
      r = 10 * r + (*ch - '0');
      io.HandleRelativePosition(1);
      ch = io.GetCurrentChar();
    } while (ch && *ch >= '0' && *ch <= '9');
    if (r > 0 && ch && *ch == '*') { // subtle: r must be nonzero
      io.HandleRelativePosition(1);
      ch = io.GetCurrentChar();
      if (ch && *ch == '/') { // r*/
        hitSlash_ = true;
        edit.descriptor = DataEdit::ListDirectedNullValue;
        return edit;
      }
      if (!ch || *ch == ' ' || *ch == '\t' || *ch == comma) { // "r*" null
        edit.descriptor = DataEdit::ListDirectedNullValue;
      }
      edit.repeat = std::min<int>(r, maxRepeat);
      remaining_ = r - edit.repeat;
      initialRecordNumber_ = connection.currentRecordNumber;
      initialPositionInRecord_ = connection.positionInRecord;
    } else { // not a repetition count, just an integer value; rewind
      connection.positionInRecord = start;
    }
  }
  if (!imaginaryPart_ && ch && *ch == '(') {
    realPart_ = true;
    io.HandleRelativePosition(1);
    edit.descriptor = DataEdit::ListDirectedRealPart;
  }
  return edit;
}

template <Direction DIR>
bool UnformattedIoStatementState<DIR>::Receive(
    char *data, std::size_t bytes, std::size_t elementBytes) {
  if constexpr (DIR == Direction::Output) {
    this->Crash(
        "UnformattedIoStatementState::Receive() called for output statement");
  }
  return this->unit().Receive(data, bytes, elementBytes, *this);
}

template <Direction DIR>
bool UnformattedIoStatementState<DIR>::Emit(
    const char *data, std::size_t bytes, std::size_t elementBytes) {
  if constexpr (DIR == Direction::Input) {
    this->Crash(
        "UnformattedIoStatementState::Emit() called for input statement");
  }
  return ExternalIoStatementState<DIR>::Emit(data, bytes, elementBytes);
}

template <Direction DIR>
int UnformattedIoStatementState<DIR>::EndIoStatement() {
  ExternalFileUnit &unit{this->unit()};
  if constexpr (DIR == Direction::Output) {
    if (unit.access == Access::Sequential && !unit.isFixedRecordLength) {
      // Append the length of a sequential unformatted variable-length record
      // as its footer, then overwrite the reserved first four bytes of the
      // record with its length as its header.  These four bytes were skipped
      // over in BeginUnformattedOutput().
      // TODO: Break very large records up into subrecords with negative
      // headers &/or footers
      union {
        std::uint32_t u;
        char c[sizeof u];
      } u;
      u.u = unit.furthestPositionInRecord - sizeof u;
      // TODO: Convert record length to little-endian on big-endian host?
      if (!(this->Emit(u.c, sizeof u) &&
              (this->HandleAbsolutePosition(0), this->Emit(u.c, sizeof u)))) {
        return false;
      }
    }
  }
  return ExternalIoStatementState<DIR>::EndIoStatement();
}

template class InternalIoStatementState<Direction::Output>;
template class InternalIoStatementState<Direction::Input>;
template class InternalFormattedIoStatementState<Direction::Output>;
template class InternalFormattedIoStatementState<Direction::Input>;
template class InternalListIoStatementState<Direction::Output>;
template class InternalListIoStatementState<Direction::Input>;
template class ExternalIoStatementState<Direction::Output>;
template class ExternalIoStatementState<Direction::Input>;
template class ExternalFormattedIoStatementState<Direction::Output>;
template class ExternalFormattedIoStatementState<Direction::Input>;
template class ExternalListIoStatementState<Direction::Output>;
template class ExternalListIoStatementState<Direction::Input>;
template class UnformattedIoStatementState<Direction::Output>;
template class UnformattedIoStatementState<Direction::Input>;

int ExternalMiscIoStatementState::EndIoStatement() {
  ExternalFileUnit &ext{unit()};
  switch (which_) {
  case Flush:
    ext.Flush(*this);
    std::fflush(nullptr); // flushes C stdio output streams (12.9(2))
    break;
  case Backspace:
    ext.BackspaceRecord(*this);
    break;
  case Endfile:
    ext.Endfile(*this);
    break;
  case Rewind:
    ext.Rewind(*this);
    break;
  }
  return ExternalIoStatementBase::EndIoStatement();
}

} // namespace Fortran::runtime::io
