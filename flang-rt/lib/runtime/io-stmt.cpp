//===-- lib/runtime/io-stmt.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/io-stmt.h"
#include "unit.h"
#include "flang-rt/runtime/connection.h"
#include "flang-rt/runtime/emit-encoded.h"
#include "flang-rt/runtime/format.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/utf.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

bool IoStatementBase::Emit(const char *, std::size_t, std::size_t) {
  return false;
}

std::size_t IoStatementBase::GetNextInputBytes(const char *&p) {
  p = nullptr;
  return 0;
}

std::size_t IoStatementBase::ViewBytesInRecord(
    const char *&p, bool forward) const {
  p = nullptr;
  return 0;
}

bool IoStatementBase::AdvanceRecord(int) { return false; }

void IoStatementBase::BackspaceRecord() {}

bool IoStatementBase::Receive(char *, std::size_t, std::size_t) {
  return false;
}

common::optional<DataEdit> IoStatementBase::GetNextDataEdit(
    IoStatementState &, int) {
  return common::nullopt;
}

bool IoStatementBase::BeginReadingRecord() { return true; }

void IoStatementBase::FinishReadingRecord() {}

void IoStatementBase::HandleAbsolutePosition(std::int64_t) {}

void IoStatementBase::HandleRelativePosition(std::int64_t) {}

std::int64_t IoStatementBase::InquirePos() { return 0; }

ExternalFileUnit *IoStatementBase::GetExternalFileUnit() const {
  return nullptr;
}

bool IoStatementBase::Inquire(InquiryKeywordHash, char *, std::size_t) {
  return false;
}

bool IoStatementBase::Inquire(InquiryKeywordHash, bool &) { return false; }

bool IoStatementBase::Inquire(InquiryKeywordHash, std::int64_t, bool &) {
  return false;
}

bool IoStatementBase::Inquire(InquiryKeywordHash, std::int64_t &) {
  return false;
}

RT_API_ATTRS static const char *InquiryKeywordHashDecode(
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

void IoStatementBase::BadInquiryKeywordHashCrash(InquiryKeywordHash inquiry) {
  char buffer[16];
  const char *decode{InquiryKeywordHashDecode(buffer, sizeof buffer, inquiry)};
  Crash("Bad InquiryKeywordHash 0x%x (%s)", inquiry,
      decode ? decode : "(cannot decode)");
}

template <Direction DIR>
InternalIoStatementState<DIR>::InternalIoStatementState(
    Buffer scalar, std::size_t length, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{scalar, length, 1} {}

template <Direction DIR>
InternalIoStatementState<DIR>::InternalIoStatementState(
    const Descriptor &d, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{d, *this} {}

template <Direction DIR>
bool InternalIoStatementState<DIR>::Emit(
    const char *data, std::size_t bytes, std::size_t /*elementBytes*/) {
  if constexpr (DIR == Direction::Input) {
    Crash("InternalIoStatementState<Direction::Input>::Emit() called");
    return false;
  }
  return unit_.Emit(data, bytes, *this);
}

template <Direction DIR>
std::size_t InternalIoStatementState<DIR>::GetNextInputBytes(const char *&p) {
  return unit_.GetNextInputBytes(p, *this);
}

// InternalIoStatementState<DIR>::ViewBytesInRecord() not needed or defined

template <Direction DIR>
bool InternalIoStatementState<DIR>::AdvanceRecord(int n) {
  while (n-- > 0) {
    if (!unit_.AdvanceRecord(*this)) {
      return false;
    }
  }
  return true;
}

template <Direction DIR> void InternalIoStatementState<DIR>::BackspaceRecord() {
  unit_.BackspaceRecord(*this);
}

template <Direction DIR> int InternalIoStatementState<DIR>::EndIoStatement() {
  auto result{IoStatementBase::EndIoStatement()};
  if (free_) {
    FreeMemory(this);
  }
  return result;
}

template <Direction DIR>
void InternalIoStatementState<DIR>::HandleAbsolutePosition(std::int64_t n) {
  return unit_.HandleAbsolutePosition(n);
}

template <Direction DIR>
void InternalIoStatementState<DIR>::HandleRelativePosition(std::int64_t n) {
  return unit_.HandleRelativePosition(n);
}

template <Direction DIR>
std::int64_t InternalIoStatementState<DIR>::InquirePos() {
  return unit_.InquirePos();
}

template <Direction DIR, typename CHAR>
RT_API_ATTRS
InternalFormattedIoStatementState<DIR, CHAR>::InternalFormattedIoStatementState(
    Buffer buffer, std::size_t length, const CharType *format,
    std::size_t formatLength, const Descriptor *formatDescriptor,
    const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR>{buffer, length, sourceFile, sourceLine},
      ioStatementState_{*this},
      format_{*this, format, formatLength, formatDescriptor} {}

template <Direction DIR, typename CHAR>
RT_API_ATTRS
InternalFormattedIoStatementState<DIR, CHAR>::InternalFormattedIoStatementState(
    const Descriptor &d, const CharType *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR>{d, sourceFile, sourceLine},
      ioStatementState_{*this},
      format_{*this, format, formatLength, formatDescriptor} {}

template <Direction DIR, typename CHAR>
void InternalFormattedIoStatementState<DIR, CHAR>::CompleteOperation() {
  if (!this->completedOperation()) {
    if constexpr (DIR == Direction::Output) {
      format_.Finish(*this);
      unit_.AdvanceRecord(*this);
    }
    IoStatementBase::CompleteOperation();
  }
}

template <Direction DIR, typename CHAR>
int InternalFormattedIoStatementState<DIR, CHAR>::EndIoStatement() {
  CompleteOperation();
  return InternalIoStatementState<DIR>::EndIoStatement();
}

template <Direction DIR>
InternalListIoStatementState<DIR>::InternalListIoStatementState(
    Buffer buffer, std::size_t length, const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR>{buffer, length, sourceFile, sourceLine},
      ioStatementState_{*this} {}

template <Direction DIR>
InternalListIoStatementState<DIR>::InternalListIoStatementState(
    const Descriptor &d, const char *sourceFile, int sourceLine)
    : InternalIoStatementState<DIR>{d, sourceFile, sourceLine},
      ioStatementState_{*this} {}

template <Direction DIR>
void InternalListIoStatementState<DIR>::CompleteOperation() {
  if (!this->completedOperation()) {
    if constexpr (DIR == Direction::Output) {
      if (unit_.furthestPositionInRecord > 0) {
        unit_.AdvanceRecord(*this);
      }
    }
    IoStatementBase::CompleteOperation();
  }
}

template <Direction DIR>
int InternalListIoStatementState<DIR>::EndIoStatement() {
  CompleteOperation();
  if constexpr (DIR == Direction::Input) {
    if (int status{ListDirectedStatementState<DIR>::EndIoStatement()};
        status != IostatOk) {
      return status;
    }
  }
  return InternalIoStatementState<DIR>::EndIoStatement();
}

ExternalIoStatementBase::ExternalIoStatementBase(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, unit_{unit} {}

MutableModes &ExternalIoStatementBase::mutableModes() {
  if (const ChildIo * child{unit_.GetChildIo()}) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
    return child->parent().mutableModes();
#else
    ReportUnsupportedChildIo();
#endif
  }
  return unit_.modes;
}

ConnectionState &ExternalIoStatementBase::GetConnectionState() { return unit_; }

int ExternalIoStatementBase::EndIoStatement() {
  CompleteOperation();
  auto result{IoStatementBase::EndIoStatement()};
#if !defined(RT_USE_PSEUDO_FILE_UNIT)
  auto unitNumber{unit_.unitNumber()};
  unit_.EndIoStatement(); // annihilates *this in unit_.u_
  if (destroy_) {
    if (ExternalFileUnit *
        toClose{ExternalFileUnit::LookUpForClose(unitNumber)}) {
      toClose->Close(CloseStatus::Delete, *this);
      toClose->DestroyClosed();
    }
  }
#else
  // Fetch the unit pointer before *this disappears.
  ExternalFileUnit *unitPtr{&unit_};
  // The pseudo file units are dynamically allocated
  // and are not tracked in the unit map.
  // They have to be destructed and deallocated here.
  unitPtr->~ExternalFileUnit();
  FreeMemory(unitPtr);
#endif
  return result;
}

void ExternalIoStatementBase::SetAsynchronous() {
  asynchronousID_ = unit().GetAsynchronousId(*this);
}

std::int64_t ExternalIoStatementBase::InquirePos() {
  return unit_.InquirePos();
}

void OpenStatementState::set_path(const char *path, std::size_t length) {
  pathLength_ = TrimTrailingSpaces(path, length);
  path_ = SaveDefaultCharacter(path, pathLength_, *this);
}

void OpenStatementState::CompleteOperation() {
  if (completedOperation()) {
    return;
  }
  if (position_) {
    if (access_ && *access_ == Access::Direct) {
      SignalError("POSITION= may not be set with ACCESS='DIRECT'");
      position_.reset();
    }
  }
  if (status_) { // 12.5.6.10
    if ((*status_ == OpenStatus::New || *status_ == OpenStatus::Replace) &&
        !path_.get()) {
      SignalError("FILE= required on OPEN with STATUS='NEW' or 'REPLACE'");
    } else if (*status_ == OpenStatus::Scratch && path_.get()) {
      SignalError("FILE= may not appear on OPEN with STATUS='SCRATCH'");
    }
  }
  // F'2023 12.5.6.13 - NEWUNIT= requires either FILE= or STATUS='SCRATCH'
  if (isNewUnit_ && !path_.get() &&
      status_.value_or(OpenStatus::Unknown) != OpenStatus::Scratch) {
    SignalError(IostatBadNewUnit);
    status_ = OpenStatus::Scratch; // error recovery
  }
  if (path_.get() || wasExtant_ ||
      (status_ && *status_ == OpenStatus::Scratch)) {
    if (unit().OpenUnit(status_, action_, position_.value_or(Position::AsIs),
            std::move(path_), pathLength_, convert_, *this)) {
      wasExtant_ = false; // existing unit was closed
    }
  } else {
    unit().OpenAnonymousUnit(
        status_, action_, position_.value_or(Position::AsIs), convert_, *this);
  }
  if (access_) {
    if (*access_ != unit().access) {
      if (wasExtant_) {
        SignalError("ACCESS= may not be changed on an open unit");
        access_.reset();
      }
    }
    if (access_) {
      unit().access = *access_;
    }
  }
  if (!unit().isUnformatted) {
    unit().isUnformatted = isUnformatted_;
  }
  if (isUnformatted_ && *isUnformatted_ != *unit().isUnformatted) {
    if (wasExtant_) {
      SignalError("FORM= may not be changed on an open unit");
    }
    unit().isUnformatted = *isUnformatted_;
  }
  if (!unit().isUnformatted) {
    // Set default format (C.7.4 point 2).
    unit().isUnformatted = unit().access != Access::Sequential;
  }
  if (unit().isUnformatted.value_or(false) && mustBeFormatted_) {
    // This is an unformatted unit, but the OPEN statement contained at least
    // one specifier that is not permitted unless the unit is formatted
    // (e.g., BLANK=).  Programs that want to detect this error (i.e., tests)
    // should be informed about it, but don't crash the program otherwise
    // since most other compilers let it slide.
    if (HasErrorRecovery()) {
      SignalError("FORM='UNFORMATTED' is not allowed with OPEN specifiers that "
                  "apply only to formatted units");
    }
  }
  if (!wasExtant_ && InError()) {
    // Release the new unit on failure
    set_destroy();
  }
  IoStatementBase::CompleteOperation();
}

int OpenStatementState::EndIoStatement() {
  CompleteOperation();
  return ExternalIoStatementBase::EndIoStatement();
}

int CloseStatementState::EndIoStatement() {
  CompleteOperation();
  int result{ExternalIoStatementBase::EndIoStatement()};
  unit().CloseUnit(status_, *this);
  unit().DestroyClosed();
  return result;
}

void NoUnitIoStatementState::CompleteOperation() {
  SignalPendingError();
  IoStatementBase::CompleteOperation();
}

int NoUnitIoStatementState::EndIoStatement() {
  CompleteOperation();
  auto result{IoStatementBase::EndIoStatement()};
  FreeMemory(this);
  return result;
}

template <Direction DIR>
ExternalIoStatementState<DIR>::ExternalIoStatementState(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
    : ExternalIoStatementBase{unit, sourceFile, sourceLine}, mutableModes_{
                                                                 unit.modes} {
  if constexpr (DIR == Direction::Output) {
    // If the last statement was a non-advancing IO input statement, the unit
    // furthestPositionInRecord was not advanced, but the positionInRecord may
    // have been advanced. Advance furthestPositionInRecord here to avoid
    // overwriting the part of the record that has been read with blanks.
    unit.furthestPositionInRecord =
        std::max(unit.furthestPositionInRecord, unit.positionInRecord);
  }
}

template <Direction DIR>
void ExternalIoStatementState<DIR>::CompleteOperation() {
  if (completedOperation()) {
    return;
  }
  if constexpr (DIR == Direction::Input) {
    BeginReadingRecord(); // in case there were no I/O items
    if (mutableModes().nonAdvancing && !InError()) {
      unit().leftTabLimit = unit().furthestPositionInRecord;
    } else {
      FinishReadingRecord();
    }
  } else { // output
    if (mutableModes().nonAdvancing) {
      // Make effects of positioning past the last Emit() visible with blanks.
      if (unit().positionInRecord > unit().furthestPositionInRecord) {
        unit().Emit("", 0, 1, *this); // Emit() will pad
      }
      unit().leftTabLimit = unit().positionInRecord;
    } else {
      unit().AdvanceRecord(*this);
    }
    unit().FlushIfTerminal(*this);
  }
  return IoStatementBase::CompleteOperation();
}

template <Direction DIR> int ExternalIoStatementState<DIR>::EndIoStatement() {
  CompleteOperation();
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
std::size_t ExternalIoStatementState<DIR>::GetNextInputBytes(const char *&p) {
  return unit().GetNextInputBytes(p, *this);
}

template <Direction DIR>
std::size_t ExternalIoStatementState<DIR>::ViewBytesInRecord(
    const char *&p, bool forward) const {
  return unit().ViewBytesInRecord(p, forward);
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

template <Direction DIR>
bool ExternalIoStatementState<DIR>::BeginReadingRecord() {
  if constexpr (DIR == Direction::Input) {
    return unit().BeginReadingRecord(*this);
  } else {
    Crash("ExternalIoStatementState<Direction::Output>::BeginReadingRecord() "
          "called");
    return false;
  }
}

template <Direction DIR>
void ExternalIoStatementState<DIR>::FinishReadingRecord() {
  if constexpr (DIR == Direction::Input) {
    unit().FinishReadingRecord(*this);
  } else {
    Crash("ExternalIoStatementState<Direction::Output>::FinishReadingRecord() "
          "called");
  }
}

template <Direction DIR, typename CHAR>
ExternalFormattedIoStatementState<DIR, CHAR>::ExternalFormattedIoStatementState(
    ExternalFileUnit &unit, const CHAR *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, const char *sourceFile, int sourceLine)
    : ExternalIoStatementState<DIR>{unit, sourceFile, sourceLine},
      format_{*this, format, formatLength, formatDescriptor} {}

template <Direction DIR, typename CHAR>
void ExternalFormattedIoStatementState<DIR, CHAR>::CompleteOperation() {
  if (this->completedOperation()) {
    return;
  }
  if constexpr (DIR == Direction::Input) {
    this->BeginReadingRecord(); // in case there were no I/O items
  }
  format_.Finish(*this);
  return ExternalIoStatementState<DIR>::CompleteOperation();
}

template <Direction DIR, typename CHAR>
int ExternalFormattedIoStatementState<DIR, CHAR>::EndIoStatement() {
  CompleteOperation();
  return ExternalIoStatementState<DIR>::EndIoStatement();
}

common::optional<DataEdit> IoStatementState::GetNextDataEdit(int n) {
  return common::visit(
      [&](auto &x) { return x.get().GetNextDataEdit(*this, n); }, u_);
}

const NonTbpDefinedIoTable *IoStatementState::nonTbpDefinedIoTable() const {
  return common::visit(
      [&](auto &x) { return x.get().nonTbpDefinedIoTable(); }, u_);
}

void IoStatementState::set_nonTbpDefinedIoTable(
    const NonTbpDefinedIoTable *table) {
  common::visit(
      [&](auto &x) { return x.get().set_nonTbpDefinedIoTable(table); }, u_);
}

bool IoStatementState::Emit(
    const char *data, std::size_t bytes, std::size_t elementBytes) {
  return common::visit(
      [=](auto &x) { return x.get().Emit(data, bytes, elementBytes); }, u_);
}

bool IoStatementState::Receive(
    char *data, std::size_t n, std::size_t elementBytes) {
  return common::visit(
      [=](auto &x) { return x.get().Receive(data, n, elementBytes); }, u_);
}

std::size_t IoStatementState::GetNextInputBytes(const char *&p) {
  return common::visit(
      [&](auto &x) { return x.get().GetNextInputBytes(p); }, u_);
}

bool IoStatementState::AdvanceRecord(int n) {
  return common::visit([=](auto &x) { return x.get().AdvanceRecord(n); }, u_);
}

void IoStatementState::BackspaceRecord() {
  common::visit([](auto &x) { x.get().BackspaceRecord(); }, u_);
}

void IoStatementState::HandleRelativePosition(std::int64_t n) {
  common::visit([=](auto &x) { x.get().HandleRelativePosition(n); }, u_);
}

void IoStatementState::HandleAbsolutePosition(std::int64_t n) {
  common::visit([=](auto &x) { x.get().HandleAbsolutePosition(n); }, u_);
}

void IoStatementState::CompleteOperation() {
  common::visit([](auto &x) { x.get().CompleteOperation(); }, u_);
}

int IoStatementState::EndIoStatement() {
  return common::visit([](auto &x) { return x.get().EndIoStatement(); }, u_);
}

ConnectionState &IoStatementState::GetConnectionState() {
  return common::visit(
      [](auto &x) -> ConnectionState & { return x.get().GetConnectionState(); },
      u_);
}

MutableModes &IoStatementState::mutableModes() {
  return common::visit(
      [](auto &x) -> MutableModes & { return x.get().mutableModes(); }, u_);
}

bool IoStatementState::BeginReadingRecord() {
  return common::visit(
      [](auto &x) { return x.get().BeginReadingRecord(); }, u_);
}

IoErrorHandler &IoStatementState::GetIoErrorHandler() const {
  return common::visit(
      [](auto &x) -> IoErrorHandler & {
        return static_cast<IoErrorHandler &>(x.get());
      },
      u_);
}

ExternalFileUnit *IoStatementState::GetExternalFileUnit() const {
  return common::visit(
      [](auto &x) { return x.get().GetExternalFileUnit(); }, u_);
}

common::optional<char32_t> IoStatementState::GetCurrentCharSlow(
    std::size_t &byteCount) {
  const char *p{nullptr};
  std::size_t bytes{GetNextInputBytes(p)};
  if (bytes == 0) {
    byteCount = 0;
    return common::nullopt;
  } else {
    const ConnectionState &connection{GetConnectionState()};
    if (connection.isUTF8) {
      std::size_t length{MeasureUTF8Bytes(*p)};
      if (length <= bytes) {
        if (auto result{DecodeUTF8(p)}) {
          byteCount = length;
          return result;
        }
      }
      GetIoErrorHandler().SignalError(IostatUTF8Decoding);
      // Error recovery: return the next byte
    } else if (connection.internalIoCharKind > 1) {
      byteCount = connection.internalIoCharKind;
      if (byteCount == 2) {
        return *reinterpret_cast<const char16_t *>(p);
      } else {
        return *reinterpret_cast<const char32_t *>(p);
      }
    }
    byteCount = 1;
    return *p;
  }
}

IoStatementState::FastAsciiField IoStatementState::GetUpcomingFastAsciiField() {
  ConnectionState &connection{GetConnectionState()};
  if (!connection.isUTF8 && connection.internalIoCharKind <= 1) {
    const char *p{nullptr};
    if (std::size_t bytes{GetNextInputBytes(p)}) {
      return FastAsciiField{connection, p, bytes};
    }
  }
  return FastAsciiField{connection};
}

common::optional<char32_t> IoStatementState::NextInField(
    common::optional<int> &remaining, const DataEdit &edit,
    FastAsciiField *field) {
  std::size_t byteCount{0};
  if (!remaining) { // Stream, list-directed, NAMELIST, &c.
    if (auto next{GetCurrentChar(byteCount, field)}) {
      if ((*next < '0' || *next == ';') && edit.width.value_or(0) == 0) {
        // list-directed, NAMELIST, I0 &c., or width-free I/G:
        // check for separator character
        switch (*next) {
        case ' ':
        case '\t':
        case '/':
        case '(':
        case ')':
        case '\'':
        case '"':
        case '*':
        case '\n': // for stream access
          return common::nullopt;
        case '&':
        case '$':
          if (edit.IsNamelist()) {
            return common::nullopt;
          }
          break;
        case ',':
          if (!(edit.modes.editingFlags & decimalComma)) {
            return common::nullopt;
          }
          break;
        case ';':
          if (edit.modes.editingFlags & decimalComma) {
            return common::nullopt;
          }
          break;
        default:
          break;
        }
      }
      if (field) {
        field->Advance(1, byteCount);
      } else {
        HandleRelativePosition(byteCount);
        GotChar(byteCount);
      }
      return next;
    }
  } else if (*remaining > 0) {
    if (auto next{GetCurrentChar(byteCount, field)}) {
      if (byteCount > static_cast<std::size_t>(*remaining)) {
        return common::nullopt;
      }
      *remaining -= byteCount;
      if (field) {
        field->Advance(1, byteCount);
      } else {
        HandleRelativePosition(byteCount);
        GotChar(byteCount);
      }
      return next;
    }
    if (CheckForEndOfRecord(0,
            field ? field->connection() : GetConnectionState())) { // do padding
      --*remaining;
      return common::optional<char32_t>{' '};
    }
  }
  return common::nullopt;
}

bool IoStatementState::CheckForEndOfRecord(
    std::size_t afterReading, const ConnectionState &connection) {
  if (!connection.IsAtEOF()) {
    if (auto length{connection.EffectiveRecordLength()}) {
      if (connection.positionInRecord +
              static_cast<std::int64_t>(afterReading) >=
          *length) {
        IoErrorHandler &handler{GetIoErrorHandler()};
        const auto &modes{mutableModes()};
        if (modes.nonAdvancing) {
          if (connection.access == Access::Stream &&
              connection.unterminatedRecord) {
            // Reading final unterminated record left by a
            // non-advancing WRITE on a stream file prior to
            // positioning or ENDFILE.
            handler.SignalEnd();
          } else {
            handler.SignalEor();
          }
        } else if (!modes.pad) {
          handler.SignalError(IostatRecordReadOverrun);
        }
        return modes.pad; // PAD='YES'
      }
    }
  }
  return false;
}

bool IoStatementState::Inquire(
    InquiryKeywordHash inquiry, char *out, std::size_t chars) {
  return common::visit(
      [&](auto &x) { return x.get().Inquire(inquiry, out, chars); }, u_);
}

bool IoStatementState::Inquire(InquiryKeywordHash inquiry, bool &out) {
  return common::visit(
      [&](auto &x) { return x.get().Inquire(inquiry, out); }, u_);
}

bool IoStatementState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t id, bool &out) {
  return common::visit(
      [&](auto &x) { return x.get().Inquire(inquiry, id, out); }, u_);
}

bool IoStatementState::Inquire(InquiryKeywordHash inquiry, std::int64_t &n) {
  return common::visit(
      [&](auto &x) { return x.get().Inquire(inquiry, n); }, u_);
}

std::int64_t IoStatementState::InquirePos() {
  return common::visit([&](auto &x) { return x.get().InquirePos(); }, u_);
}

void IoStatementState::GotChar(int n) {
  if (auto *formattedIn{
          get_if<FormattedIoStatementState<Direction::Input>>()}) {
    formattedIn->GotChar(n);
  } else {
    GetIoErrorHandler().Crash("IoStatementState::GotChar() called for "
                              "statement that is not formatted input");
  }
}

std::size_t
FormattedIoStatementState<Direction::Input>::GetEditDescriptorChars() const {
  return chars_;
}

void FormattedIoStatementState<Direction::Input>::GotChar(int n) {
  chars_ += n;
}

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
    return EmitAscii(io, " ", 1);
  }
  return true;
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

int ListDirectedStatementState<Direction::Input>::EndIoStatement() {
  if (repeatPosition_) {
    repeatPosition_->Cancel();
  }
  return IostatOk;
}

common::optional<DataEdit>
ListDirectedStatementState<Direction::Input>::GetNextDataEdit(
    IoStatementState &io, int maxRepeat) {
  // N.B. list-directed transfers cannot be nonadvancing (C1221)
  DataEdit edit;
  edit.descriptor = DataEdit::ListDirected;
  edit.repeat = 1; // may be overridden below
  edit.modes = io.mutableModes();
  if (hitSlash_) { // everything after '/' is nullified
    edit.descriptor = DataEdit::ListDirectedNullValue;
    return edit;
  }
  const char32_t comma{edit.modes.GetSeparatorChar()};
  std::size_t byteCount{0};
  if (remaining_ > 0 && !realPart_) { // "r*c" repetition in progress
    RUNTIME_CHECK(io.GetIoErrorHandler(), repeatPosition_.has_value());
    repeatPosition_.reset(); // restores the saved position
    if (!imaginaryPart_) {
      edit.repeat = std::min<int>(remaining_, maxRepeat);
      auto ch{io.GetCurrentChar(byteCount)};
      if (!ch || *ch == ' ' || *ch == '\t' || *ch == comma) {
        // "r*" repeated null
        edit.descriptor = DataEdit::ListDirectedNullValue;
      }
    }
    remaining_ -= edit.repeat;
    if (remaining_ > 0) {
      repeatPosition_.emplace(io);
    }
    if (!imaginaryPart_) {
      return edit;
    }
  }
  // Skip separators, handle a "r*c" repeat count; see 13.10.2 in Fortran 2018
  if (imaginaryPart_) {
    imaginaryPart_ = false;
  } else if (realPart_) {
    realPart_ = false;
    imaginaryPart_ = true;
    edit.descriptor = DataEdit::ListDirectedImaginaryPart;
  }
  auto fastField{io.GetUpcomingFastAsciiField()};
  auto ch{io.GetNextNonBlank(byteCount, &fastField)};
  if (ch && *ch == comma && eatComma_) {
    // Consume comma & whitespace after previous item.
    // This includes the comma between real and imaginary components
    // in list-directed/NAMELIST complex input.
    // (When DECIMAL='COMMA', the comma is actually a semicolon.)
    fastField.Advance(0, byteCount);
    ch = io.GetNextNonBlank(byteCount, &fastField);
  }
  eatComma_ = true;
  if (!ch) {
    return common::nullopt;
  }
  if (*ch == '/') {
    hitSlash_ = true;
    edit.descriptor = DataEdit::ListDirectedNullValue;
    return edit;
  }
  if (*ch == comma) { // separator: null value
    edit.descriptor = DataEdit::ListDirectedNullValue;
    return edit;
  }
  if (imaginaryPart_) { // can't repeat components
    return edit;
  }
  if (*ch >= '0' && *ch <= '9' && fastField.MightHaveAsterisk()) {
    // look for "r*" repetition count
    auto start{fastField.connection().positionInRecord};
    int r{0};
    do {
      static auto constexpr clamp{(std::numeric_limits<int>::max() - '9') / 10};
      if (r >= clamp) {
        r = 0;
        break;
      }
      r = 10 * r + (*ch - '0');
      fastField.Advance(0, byteCount);
      ch = io.GetCurrentChar(byteCount, &fastField);
    } while (ch && *ch >= '0' && *ch <= '9');
    if (r > 0 && ch && *ch == '*') { // subtle: r must be nonzero
      fastField.Advance(0, byteCount);
      ch = io.GetCurrentChar(byteCount, &fastField);
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
      if (remaining_ > 0) {
        repeatPosition_.emplace(io);
      }
    } else { // not a repetition count, just an integer value; rewind
      fastField.connection().positionInRecord = start;
    }
  }
  if (!imaginaryPart_ && edit.descriptor == DataEdit::ListDirected && ch &&
      *ch == '(') {
    if (maxRepeat > 0) { // not being peeked at fram DefinedFormattedIo()
      realPart_ = true;
      fastField.connection().HandleRelativePosition(byteCount);
    }
    edit.descriptor = DataEdit::ListDirectedRealPart;
  }
  return edit;
}

template <Direction DIR>
int ExternalListIoStatementState<DIR>::EndIoStatement() {
  if constexpr (DIR == Direction::Input) {
    if (auto status{ListDirectedStatementState<DIR>::EndIoStatement()};
        status != IostatOk) {
      return status;
    }
  }
  return ExternalIoStatementState<DIR>::EndIoStatement();
}

template <Direction DIR>
bool ExternalUnformattedIoStatementState<DIR>::Receive(
    char *data, std::size_t bytes, std::size_t elementBytes) {
  if constexpr (DIR == Direction::Output) {
    this->Crash("ExternalUnformattedIoStatementState::Receive() called for "
                "output statement");
  }
  return this->unit().Receive(data, bytes, elementBytes, *this);
}

template <Direction DIR>
ChildIoStatementState<DIR>::ChildIoStatementState(
    ChildIo &child, const char *sourceFile, int sourceLine)
    : IoStatementBase{sourceFile, sourceLine}, child_{child},
      mutableModes_{child.parent().mutableModes()} {}

template <Direction DIR>
const NonTbpDefinedIoTable *
ChildIoStatementState<DIR>::nonTbpDefinedIoTable() const {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().nonTbpDefinedIoTable();
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
void ChildIoStatementState<DIR>::set_nonTbpDefinedIoTable(
    const NonTbpDefinedIoTable *table) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  child_.parent().set_nonTbpDefinedIoTable(table);
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
ConnectionState &ChildIoStatementState<DIR>::GetConnectionState() {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().GetConnectionState();
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
ExternalFileUnit *ChildIoStatementState<DIR>::GetExternalFileUnit() const {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().GetExternalFileUnit();
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR> int ChildIoStatementState<DIR>::EndIoStatement() {
  CompleteOperation();
  auto result{IoStatementBase::EndIoStatement()};
  child_.EndIoStatement(); // annihilates *this in child_.u_
  return result;
}

template <Direction DIR>
bool ChildIoStatementState<DIR>::Emit(
    const char *data, std::size_t bytes, std::size_t elementBytes) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().Emit(data, bytes, elementBytes);
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
std::size_t ChildIoStatementState<DIR>::GetNextInputBytes(const char *&p) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().GetNextInputBytes(p);
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
void ChildIoStatementState<DIR>::HandleAbsolutePosition(std::int64_t n) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().HandleAbsolutePosition(n);
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
void ChildIoStatementState<DIR>::HandleRelativePosition(std::int64_t n) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return child_.parent().HandleRelativePosition(n);
#else
  ReportUnsupportedChildIo();
#endif
}

template <Direction DIR, typename CHAR>
ChildFormattedIoStatementState<DIR, CHAR>::ChildFormattedIoStatementState(
    ChildIo &child, const CHAR *format, std::size_t formatLength,
    const Descriptor *formatDescriptor, const char *sourceFile, int sourceLine)
    : ChildIoStatementState<DIR>{child, sourceFile, sourceLine},
      format_{*this, format, formatLength, formatDescriptor} {}

template <Direction DIR, typename CHAR>
void ChildFormattedIoStatementState<DIR, CHAR>::CompleteOperation() {
  if (!this->completedOperation()) {
    format_.Finish(*this);
    ChildIoStatementState<DIR>::CompleteOperation();
  }
}

template <Direction DIR, typename CHAR>
int ChildFormattedIoStatementState<DIR, CHAR>::EndIoStatement() {
  CompleteOperation();
  return ChildIoStatementState<DIR>::EndIoStatement();
}

template <Direction DIR, typename CHAR>
bool ChildFormattedIoStatementState<DIR, CHAR>::AdvanceRecord(int n) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return this->child().parent().AdvanceRecord(n);
#else
  this->ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
ChildListIoStatementState<DIR>::ChildListIoStatementState(
    ChildIo &child, const char *sourceFile, int sourceLine)
    : ChildIoStatementState<DIR>{child, sourceFile, sourceLine} {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  if constexpr (DIR == Direction::Input) {
    if (auto *listInput{child.parent()
                .get_if<ListDirectedStatementState<Direction::Input>>()}) {
      this->namelistGroup_ = listInput->namelistGroup();
    }
  }
#else
  this->ReportUnsupportedChildIo();
#endif
}

template <Direction DIR>
bool ChildListIoStatementState<DIR>::AdvanceRecord(int n) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  // Allow child NAMELIST input to advance
  if (DIR == Direction::Input && this->mutableModes().inNamelist) {
    return this->child().parent().AdvanceRecord(n);
  } else {
    return false;
  }
#else
  this->ReportUnsupportedChildIo();
#endif
}

template <Direction DIR> int ChildListIoStatementState<DIR>::EndIoStatement() {
  if constexpr (DIR == Direction::Input) {
    if (int status{ListDirectedStatementState<DIR>::EndIoStatement()};
        status != IostatOk) {
      return status;
    }
  }
  return ChildIoStatementState<DIR>::EndIoStatement();
}

template <Direction DIR>
bool ChildUnformattedIoStatementState<DIR>::Receive(
    char *data, std::size_t bytes, std::size_t elementBytes) {
#if !defined(RT_DEVICE_AVOID_RECURSION)
  return this->child().parent().Receive(data, bytes, elementBytes);
#else
  this->ReportUnsupportedChildIo();
#endif
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
template class ExternalUnformattedIoStatementState<Direction::Output>;
template class ExternalUnformattedIoStatementState<Direction::Input>;
template class ChildIoStatementState<Direction::Output>;
template class ChildIoStatementState<Direction::Input>;
template class ChildFormattedIoStatementState<Direction::Output>;
template class ChildFormattedIoStatementState<Direction::Input>;
template class ChildListIoStatementState<Direction::Output>;
template class ChildListIoStatementState<Direction::Input>;
template class ChildUnformattedIoStatementState<Direction::Output>;
template class ChildUnformattedIoStatementState<Direction::Input>;

void ExternalMiscIoStatementState::CompleteOperation() {
  if (completedOperation()) {
    return;
  }
  ExternalFileUnit &ext{unit()};
  switch (which_) {
  case Flush:
    ext.FlushOutput(*this);
#if !defined(RT_DEVICE_COMPILATION)
    std::fflush(nullptr); // flushes C stdio output streams (12.9(2))
#endif
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
  case Wait:
    break; // handled in io-api.cpp BeginWait
  }
  return IoStatementBase::CompleteOperation();
}

int ExternalMiscIoStatementState::EndIoStatement() {
  CompleteOperation();
  return ExternalIoStatementBase::EndIoStatement();
}

InquireUnitState::InquireUnitState(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
    : ExternalIoStatementBase{unit, sourceFile, sourceLine} {}

bool InquireUnitState::Inquire(
    InquiryKeywordHash inquiry, char *result, std::size_t length) {
  if (unit().createdForInternalChildIo()) {
    SignalError(IostatInquireInternalUnit,
        "INQUIRE of unit created for defined derived type I/O of an internal "
        "unit");
    return false;
  }
  const char *str{nullptr};
  switch (inquiry) {
  case HashInquiryKeyword("ACCESS"):
    if (!unit().IsConnected()) {
      str = "UNDEFINED";
    } else {
      switch (unit().access) {
      case Access::Sequential:
        str = "SEQUENTIAL";
        break;
      case Access::Direct:
        str = "DIRECT";
        break;
      case Access::Stream:
        str = "STREAM";
        break;
      }
    }
    break;
  case HashInquiryKeyword("ACTION"):
    str = !unit().IsConnected() ? "UNDEFINED"
        : unit().mayWrite()     ? unit().mayRead() ? "READWRITE" : "WRITE"
                                : "READ";
    break;
  case HashInquiryKeyword("ASYNCHRONOUS"):
    str = !unit().IsConnected()    ? "UNDEFINED"
        : unit().mayAsynchronous() ? "YES"
                                   : "NO";
    break;
  case HashInquiryKeyword("BLANK"):
    str = !unit().IsConnected() || unit().isUnformatted.value_or(true)
        ? "UNDEFINED"
        : mutableModes().editingFlags & blankZero ? "ZERO"
                                                  : "NULL";
    break;
  case HashInquiryKeyword("CARRIAGECONTROL"):
    str = "LIST";
    break;
  case HashInquiryKeyword("CONVERT"):
    str = unit().swapEndianness() ? "SWAP" : "NATIVE";
    break;
  case HashInquiryKeyword("DECIMAL"):
    str = !unit().IsConnected() || unit().isUnformatted.value_or(true)
        ? "UNDEFINED"
        : mutableModes().editingFlags & decimalComma ? "COMMA"
                                                     : "POINT";
    break;
  case HashInquiryKeyword("DELIM"):
    if (!unit().IsConnected() || unit().isUnformatted.value_or(true)) {
      str = "UNDEFINED";
    } else {
      switch (mutableModes().delim) {
      case '\'':
        str = "APOSTROPHE";
        break;
      case '"':
        str = "QUOTE";
        break;
      default:
        str = "NONE";
        break;
      }
    }
    break;
  case HashInquiryKeyword("DIRECT"):
    str = !unit().IsConnected() ? "UNKNOWN"
        : unit().access == Access::Direct ||
            (unit().mayPosition() && unit().openRecl)
        ? "YES"
        : "NO";
    break;
  case HashInquiryKeyword("ENCODING"):
    str = !unit().IsConnected()               ? "UNKNOWN"
        : unit().isUnformatted.value_or(true) ? "UNDEFINED"
        : unit().isUTF8                       ? "UTF-8"
                                              : "ASCII";
    break;
  case HashInquiryKeyword("FORM"):
    str = !unit().IsConnected() || !unit().isUnformatted ? "UNDEFINED"
        : *unit().isUnformatted                          ? "UNFORMATTED"
                                                         : "FORMATTED";
    break;
  case HashInquiryKeyword("FORMATTED"):
    str = !unit().IsConnected() ? "UNDEFINED"
        : !unit().isUnformatted ? "UNKNOWN"
        : *unit().isUnformatted ? "NO"
                                : "YES";
    break;
  case HashInquiryKeyword("NAME"):
    str = unit().path();
    if (!str) {
      return true; // result is undefined
    }
    break;
  case HashInquiryKeyword("PAD"):
    str = !unit().IsConnected() || unit().isUnformatted.value_or(true)
        ? "UNDEFINED"
        : mutableModes().pad ? "YES"
                             : "NO";
    break;
  case HashInquiryKeyword("POSITION"):
    if (!unit().IsConnected() || unit().access == Access::Direct) {
      str = "UNDEFINED";
    } else {
      switch (unit().InquirePosition()) {
      case Position::Rewind:
        str = "REWIND";
        break;
      case Position::Append:
        str = "APPEND";
        break;
      case Position::AsIs:
        str = "ASIS";
        break;
      }
    }
    break;
  case HashInquiryKeyword("READ"):
    str = !unit().IsConnected() ? "UNDEFINED" : unit().mayRead() ? "YES" : "NO";
    break;
  case HashInquiryKeyword("READWRITE"):
    str = !unit().IsConnected()                 ? "UNDEFINED"
        : unit().mayRead() && unit().mayWrite() ? "YES"
                                                : "NO";
    break;
  case HashInquiryKeyword("ROUND"):
    if (!unit().IsConnected() || unit().isUnformatted.value_or(true)) {
      str = "UNDEFINED";
    } else {
      switch (mutableModes().round) {
      case decimal::FortranRounding::RoundNearest:
        str = "NEAREST";
        break;
      case decimal::FortranRounding::RoundUp:
        str = "UP";
        break;
      case decimal::FortranRounding::RoundDown:
        str = "DOWN";
        break;
      case decimal::FortranRounding::RoundToZero:
        str = "ZERO";
        break;
      case decimal::FortranRounding::RoundCompatible:
        str = "COMPATIBLE";
        break;
      }
    }
    break;
  case HashInquiryKeyword("SEQUENTIAL"):
    // "NO" for Direct, since Sequential would not work if
    // the unit were reopened without RECL=.
    str = !unit().IsConnected()               ? "UNKNOWN"
        : unit().access == Access::Sequential ? "YES"
                                              : "NO";
    break;
  case HashInquiryKeyword("SIGN"):
    str = !unit().IsConnected() || unit().isUnformatted.value_or(true)
        ? "UNDEFINED"
        : mutableModes().editingFlags & signPlus ? "PLUS"
                                                 : "SUPPRESS";
    break;
  case HashInquiryKeyword("STREAM"):
    str = !unit().IsConnected()           ? "UNKNOWN"
        : unit().access == Access::Stream ? "YES"
                                          : "NO";
    break;
  case HashInquiryKeyword("UNFORMATTED"):
    str = !unit().IsConnected() || !unit().isUnformatted ? "UNKNOWN"
        : *unit().isUnformatted                          ? "YES"
                                                         : "NO";
    break;
  case HashInquiryKeyword("WRITE"):
    str = !unit().IsConnected() ? "UNKNOWN" : unit().mayWrite() ? "YES" : "NO";
    break;
  }
  if (str) {
    ToFortranDefaultCharacter(result, length, str);
    return true;
  } else {
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireUnitState::Inquire(InquiryKeywordHash inquiry, bool &result) {
  switch (inquiry) {
  case HashInquiryKeyword("EXIST"):
    result = true;
    return true;
  case HashInquiryKeyword("NAMED"):
    result = unit().path() != nullptr;
    return true;
  case HashInquiryKeyword("OPENED"):
    result = unit().IsConnected();
    return true;
  case HashInquiryKeyword("PENDING"):
    result = false; // asynchronous I/O is not implemented
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireUnitState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t, bool &result) {
  switch (inquiry) {
  case HashInquiryKeyword("PENDING"):
    result = false; // asynchronous I/O is not implemented
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireUnitState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t &result) {
  switch (inquiry) {
  case HashInquiryKeyword("NEXTREC"):
    if (unit().access == Access::Direct) {
      result = unit().currentRecordNumber;
    }
    return true;
  case HashInquiryKeyword("NUMBER"):
    result = unit().unitNumber();
    return true;
  case HashInquiryKeyword("POS"):
    result = unit().InquirePos();
    return true;
  case HashInquiryKeyword("RECL"):
    if (!unit().IsConnected()) {
      result = -1;
    } else if (unit().access == Access::Stream) {
      result = -2;
    } else if (unit().openRecl) {
      result = *unit().openRecl;
    } else {
      result = std::numeric_limits<std::int32_t>::max();
    }
    return true;
  case HashInquiryKeyword("SIZE"):
    result = -1;
    if (unit().IsConnected()) {
      unit().FlushOutput(*this);
      if (auto size{unit().knownSize()}) {
        result = *size;
      }
    }
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

InquireNoUnitState::InquireNoUnitState(
    const char *sourceFile, int sourceLine, int badUnitNumber)
    : NoUnitIoStatementState{*this, sourceFile, sourceLine, badUnitNumber} {}

bool InquireNoUnitState::Inquire(
    InquiryKeywordHash inquiry, char *result, std::size_t length) {
  switch (inquiry) {
  case HashInquiryKeyword("ACCESS"):
  case HashInquiryKeyword("ACTION"):
  case HashInquiryKeyword("ASYNCHRONOUS"):
  case HashInquiryKeyword("BLANK"):
  case HashInquiryKeyword("CARRIAGECONTROL"):
  case HashInquiryKeyword("CONVERT"):
  case HashInquiryKeyword("DECIMAL"):
  case HashInquiryKeyword("DELIM"):
  case HashInquiryKeyword("FORM"):
  case HashInquiryKeyword("NAME"):
  case HashInquiryKeyword("PAD"):
  case HashInquiryKeyword("POSITION"):
  case HashInquiryKeyword("ROUND"):
  case HashInquiryKeyword("SIGN"):
    ToFortranDefaultCharacter(result, length, "UNDEFINED");
    return true;
  case HashInquiryKeyword("DIRECT"):
  case HashInquiryKeyword("ENCODING"):
  case HashInquiryKeyword("FORMATTED"):
  case HashInquiryKeyword("READ"):
  case HashInquiryKeyword("READWRITE"):
  case HashInquiryKeyword("SEQUENTIAL"):
  case HashInquiryKeyword("STREAM"):
  case HashInquiryKeyword("WRITE"):
  case HashInquiryKeyword("UNFORMATTED"):
    ToFortranDefaultCharacter(result, length, "UNKNOWN");
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireNoUnitState::Inquire(InquiryKeywordHash inquiry, bool &result) {
  switch (inquiry) {
  case HashInquiryKeyword("EXIST"):
    result = badUnitNumber() >= 0;
    return true;
  case HashInquiryKeyword("NAMED"):
  case HashInquiryKeyword("OPENED"):
  case HashInquiryKeyword("PENDING"):
    result = false;
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireNoUnitState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t, bool &result) {
  switch (inquiry) {
  case HashInquiryKeyword("PENDING"):
    result = false;
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireNoUnitState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t &result) {
  switch (inquiry) {
  case HashInquiryKeyword("NUMBER"):
    result = badUnitNumber();
    return true;
  case HashInquiryKeyword("NEXTREC"):
  case HashInquiryKeyword("POS"):
  case HashInquiryKeyword("RECL"):
  case HashInquiryKeyword("SIZE"):
    result = -1;
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

InquireUnconnectedFileState::InquireUnconnectedFileState(
    OwningPtr<char> &&path, const char *sourceFile, int sourceLine)
    : NoUnitIoStatementState{*this, sourceFile, sourceLine}, path_{std::move(
                                                                 path)} {}

bool InquireUnconnectedFileState::Inquire(
    InquiryKeywordHash inquiry, char *result, std::size_t length) {
  const char *str{nullptr};
  switch (inquiry) {
  case HashInquiryKeyword("ACCESS"):
  case HashInquiryKeyword("ACTION"):
  case HashInquiryKeyword("ASYNCHRONOUS"):
  case HashInquiryKeyword("BLANK"):
  case HashInquiryKeyword("CARRIAGECONTROL"):
  case HashInquiryKeyword("CONVERT"):
  case HashInquiryKeyword("DECIMAL"):
  case HashInquiryKeyword("DELIM"):
  case HashInquiryKeyword("FORM"):
  case HashInquiryKeyword("PAD"):
  case HashInquiryKeyword("POSITION"):
  case HashInquiryKeyword("ROUND"):
  case HashInquiryKeyword("SIGN"):
    str = "UNDEFINED";
    break;
  case HashInquiryKeyword("DIRECT"):
  case HashInquiryKeyword("ENCODING"):
  case HashInquiryKeyword("FORMATTED"):
  case HashInquiryKeyword("SEQUENTIAL"):
  case HashInquiryKeyword("STREAM"):
  case HashInquiryKeyword("UNFORMATTED"):
    str = "UNKNOWN";
    break;
  case HashInquiryKeyword("READ"):
    str =
        IsExtant(path_.get()) ? MayRead(path_.get()) ? "YES" : "NO" : "UNKNOWN";
    break;
  case HashInquiryKeyword("READWRITE"):
    str = IsExtant(path_.get()) ? MayReadAndWrite(path_.get()) ? "YES" : "NO"
                                : "UNKNOWN";
    break;
  case HashInquiryKeyword("WRITE"):
    str = IsExtant(path_.get()) ? MayWrite(path_.get()) ? "YES" : "NO"
                                : "UNKNOWN";
    break;
  case HashInquiryKeyword("NAME"):
    str = path_.get();
    if (!str) {
      return true; // result is undefined
    }
    break;
  }
  if (str) {
    ToFortranDefaultCharacter(result, length, str);
    return true;
  } else {
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireUnconnectedFileState::Inquire(
    InquiryKeywordHash inquiry, bool &result) {
  switch (inquiry) {
  case HashInquiryKeyword("EXIST"):
    result = IsExtant(path_.get());
    return true;
  case HashInquiryKeyword("NAMED"):
    result = true;
    return true;
  case HashInquiryKeyword("OPENED"):
    result = false;
    return true;
  case HashInquiryKeyword("PENDING"):
    result = false;
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireUnconnectedFileState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t, bool &result) {
  switch (inquiry) {
  case HashInquiryKeyword("PENDING"):
    result = false;
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

bool InquireUnconnectedFileState::Inquire(
    InquiryKeywordHash inquiry, std::int64_t &result) {
  switch (inquiry) {
  case HashInquiryKeyword("NEXTREC"):
  case HashInquiryKeyword("NUMBER"):
  case HashInquiryKeyword("POS"):
  case HashInquiryKeyword("RECL"):
    result = -1;
    return true;
  case HashInquiryKeyword("SIZE"):
    result = SizeInBytes(path_.get());
    return true;
  default:
    BadInquiryKeywordHashCrash(inquiry);
    return false;
  }
}

InquireIOLengthState::InquireIOLengthState(
    const char *sourceFile, int sourceLine)
    : NoUnitIoStatementState{*this, sourceFile, sourceLine} {}

bool InquireIOLengthState::Emit(const char *, std::size_t bytes, std::size_t) {
  bytes_ += bytes;
  return true;
}

int ErroneousIoStatementState::EndIoStatement() {
  SignalPendingError();
  if (unit_) {
    unit_->EndIoStatement();
  }
  return IoStatementBase::EndIoStatement();
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
