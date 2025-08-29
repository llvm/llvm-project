//===-- include/flang-rt/runtime/io-stmt.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Representations of the state of an I/O statement in progress

#ifndef FLANG_RT_RUNTIME_IO_STMT_H_
#define FLANG_RT_RUNTIME_IO_STMT_H_

#include "connection.h"
#include "descriptor.h"
#include "file.h"
#include "format.h"
#include "internal-unit.h"
#include "io-error.h"
#include "flang/Common/optional.h"
#include "flang/Common/reference-wrapper.h"
#include "flang/Common/visit.h"
#include "flang/Runtime/freestanding-tools.h"
#include "flang/Runtime/io-api.h"
#include <flang/Common/variant.h>
#include <functional>
#include <type_traits>

namespace Fortran::runtime::io {

RT_OFFLOAD_API_GROUP_BEGIN

class ExternalFileUnit;
class ChildIo;

class OpenStatementState;
class InquireUnitState;
class InquireNoUnitState;
class InquireUnconnectedFileState;
class InquireIOLengthState;
class ExternalMiscIoStatementState;
class CloseStatementState;
class NoopStatementState; // CLOSE or FLUSH on unknown unit
class ErroneousIoStatementState;

template <Direction, typename CHAR = char>
class InternalFormattedIoStatementState;
template <Direction> class InternalListIoStatementState;
template <Direction, typename CHAR = char>
class ExternalFormattedIoStatementState;
template <Direction> class ExternalListIoStatementState;
template <Direction> class ExternalUnformattedIoStatementState;
template <Direction, typename CHAR = char> class ChildFormattedIoStatementState;
template <Direction> class ChildListIoStatementState;
template <Direction> class ChildUnformattedIoStatementState;

struct InputStatementState {};
struct OutputStatementState {};
template <Direction D>
using IoDirectionState = std::conditional_t<D == Direction::Input,
    InputStatementState, OutputStatementState>;

// Common state for all kinds of formatted I/O
template <Direction D> class FormattedIoStatementState {};
template <> class FormattedIoStatementState<Direction::Input> {
public:
  RT_API_ATTRS std::size_t GetEditDescriptorChars() const;
  RT_API_ATTRS void GotChar(int);

private:
  // Account of characters read for edit descriptors (i.e., formatted I/O
  // with a FORMAT, not list-directed or NAMELIST), not including padding.
  std::size_t chars_{0}; // for READ(SIZE=)
};

// The Cookie type in the I/O API is a pointer (for C) to this class.
class IoStatementState {
public:
  template <typename A> explicit RT_API_ATTRS IoStatementState(A &x) : u_{x} {}

  // These member functions each project themselves into the active alternative.
  // They're used by per-data-item routines in the I/O API (e.g., OutputReal64)
  // to interact with the state of the I/O statement in progress.
  // This design avoids virtual member functions and function pointers,
  // which may not have good support in some runtime environments.

  RT_API_ATTRS const NonTbpDefinedIoTable *nonTbpDefinedIoTable() const;
  RT_API_ATTRS void set_nonTbpDefinedIoTable(const NonTbpDefinedIoTable *);

  // CompleteOperation() is the last opportunity to raise an I/O error.
  // It is called by EndIoStatement(), but it can be invoked earlier to
  // catch errors for (e.g.) GetIoMsg() and GetNewUnit().  If called
  // more than once, it is a no-op.
  RT_API_ATTRS void CompleteOperation();
  // Completes an I/O statement and reclaims storage.
  RT_API_ATTRS int EndIoStatement();

  RT_API_ATTRS bool Emit(
      const char *, std::size_t bytes, std::size_t elementBytes = 0);
  RT_API_ATTRS bool Receive(char *, std::size_t, std::size_t elementBytes = 0);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&);
  RT_API_ATTRS std::size_t ViewBytesInRecord(const char *&, bool forward) const;
  RT_API_ATTRS bool AdvanceRecord(int = 1);
  RT_API_ATTRS void BackspaceRecord();
  RT_API_ATTRS void HandleRelativePosition(std::int64_t byteOffset);
  RT_API_ATTRS void HandleAbsolutePosition(
      std::int64_t byteOffset); // for r* in list I/O
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(int maxRepeat = 1);
  RT_API_ATTRS ExternalFileUnit *
  GetExternalFileUnit() const; // null if internal unit
  RT_API_ATTRS bool BeginReadingRecord();
  RT_API_ATTRS void FinishReadingRecord();
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, char *, std::size_t);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, bool &);
  RT_API_ATTRS bool Inquire(
      InquiryKeywordHash, std::int64_t, bool &); // PENDING=
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t &);
  RT_API_ATTRS std::int64_t InquirePos();
  RT_API_ATTRS void GotChar(signed int = 1); // for READ(SIZE=); can be <0

  RT_API_ATTRS MutableModes &mutableModes();
  RT_API_ATTRS ConnectionState &GetConnectionState();
  RT_API_ATTRS IoErrorHandler &GetIoErrorHandler() const;

  // N.B.: this also works with base classes
  template <typename A> RT_API_ATTRS A *get_if() const {
    return common::visit(
        [](auto &x) -> A * {
          if constexpr (std::is_convertible_v<decltype(x.get()), A &>) {
            return &x.get();
          }
          return nullptr;
        },
        u_);
  }

  // Vacant after the end of the current record
  RT_API_ATTRS common::optional<char32_t> GetCurrentCharSlow(
      std::size_t &byteCount);

  // For faster formatted input editing, this structure can be built by
  // GetUpcomingFastAsciiField() and used to save significant time in
  // GetCurrentChar, NextInField() and other input utilities when the input
  // is buffered, does not require UTF-8 conversion, and comprises only
  // single byte characters.
  class FastAsciiField {
  public:
    RT_API_ATTRS FastAsciiField(ConnectionState &connection)
        : connection_{connection} {}
    RT_API_ATTRS FastAsciiField(
        ConnectionState &connection, const char *start, std::size_t bytes)
        : connection_{connection}, at_{start}, limit_{start + bytes} {
      CheckForAsterisk();
    }
    RT_API_ATTRS ConnectionState &connection() { return connection_; }
    RT_API_ATTRS std::size_t got() const { return got_; }

    RT_API_ATTRS bool MustUseSlowPath() const { return at_ == nullptr; }

    RT_API_ATTRS common::optional<char32_t> Next() const {
      if (at_ && at_ < limit_) {
        return *at_;
      } else {
        return common::nullopt;
      }
    }
    RT_API_ATTRS void NextRecord(IoStatementState &io) {
      if (at_) {
        if (std::size_t bytes{io.GetNextInputBytes(at_)}) {
          limit_ = at_ + bytes;
          CheckForAsterisk();
        } else {
          at_ = limit_ = nullptr;
        }
      }
    }
    RT_API_ATTRS void Advance(int gotten, std::size_t bytes) {
      if (at_ && at_ < limit_) {
        ++at_;
        got_ += gotten;
      }
      connection_.HandleRelativePosition(bytes);
    }
    RT_API_ATTRS bool MightHaveAsterisk() const { return !at_ || hasAsterisk_; }

  private:
    RT_API_ATTRS void CheckForAsterisk() {
      hasAsterisk_ = at_ && at_ < limit_ &&
          runtime::memchr(at_, '*', limit_ - at_) != nullptr;
    }

    ConnectionState &connection_;
    const char *at_{nullptr};
    const char *limit_{nullptr};
    std::size_t got_{0}; // for READ(..., SIZE=)
    bool hasAsterisk_{false};
  };

  RT_API_ATTRS FastAsciiField GetUpcomingFastAsciiField();

  RT_API_ATTRS common::optional<char32_t> GetCurrentChar(
      std::size_t &byteCount, FastAsciiField *field = nullptr) {
    if (field) {
      if (auto ch{field->Next()}) {
        byteCount = ch ? 1 : 0;
        return ch;
      } else if (!field->MustUseSlowPath()) {
        return common::nullopt;
      }
    }
    return GetCurrentCharSlow(byteCount);
  }

  // The result of CueUpInput() and the "remaining" arguments to SkipSpaces()
  // and NextInField() are always in units of bytes, not characters; the
  // distinction matters for internal input from CHARACTER(KIND=2 and 4).

  // For fixed-width fields, return the number of remaining bytes.
  // Skip over leading blanks.
  RT_API_ATTRS common::optional<int> CueUpInput(
      const DataEdit &edit, FastAsciiField *fastField = nullptr) {
    common::optional<int> remaining;
    if (edit.IsListDirected()) {
      std::size_t byteCount{0};
      GetNextNonBlank(byteCount, fastField);
    } else {
      if (edit.width.value_or(0) > 0) {
        remaining = *edit.width;
        if (int bytesPerChar{GetConnectionState().internalIoCharKind};
            bytesPerChar > 1) {
          *remaining *= bytesPerChar;
        }
      }
      SkipSpaces(remaining, fastField);
    }
    return remaining;
  }

  RT_API_ATTRS common::optional<char32_t> SkipSpaces(
      common::optional<int> &remaining, FastAsciiField *fastField = nullptr) {
    while (!remaining || *remaining > 0) {
      std::size_t byteCount{0};
      if (auto ch{GetCurrentChar(byteCount, fastField)}) {
        if (*ch != ' ' && *ch != '\t') {
          return ch;
        }
        if (remaining) {
          if (static_cast<std::size_t>(*remaining) < byteCount) {
            break;
          }
          GotChar(byteCount);
          *remaining -= byteCount;
        }
        if (fastField) {
          fastField->Advance(0, byteCount);
        } else {
          HandleRelativePosition(byteCount);
        }
      } else {
        break;
      }
    }
    return common::nullopt;
  }

  // Acquires the next input character, respecting any applicable field width
  // or separator character.
  RT_API_ATTRS common::optional<char32_t> NextInField(
      common::optional<int> &remaining, const DataEdit &,
      FastAsciiField *field = nullptr);

  // Detect and signal any end-of-record condition after input.
  // Returns true if at EOR and remaining input should be padded with blanks.
  RT_API_ATTRS bool CheckForEndOfRecord(
      std::size_t afterReading, const ConnectionState &);

  // Skips spaces, advances records, and ignores NAMELIST comments
  RT_API_ATTRS common::optional<char32_t> GetNextNonBlank(
      std::size_t &byteCount, FastAsciiField *fastField = nullptr) {
    auto ch{GetCurrentChar(byteCount, fastField)};
    bool inNamelist{mutableModes().inNamelist};
    while (!ch || *ch == ' ' || *ch == '\t' || *ch == '\n' ||
        (inNamelist && *ch == '!')) {
      if (ch && (*ch == ' ' || *ch == '\t' || *ch == '\n')) {
        if (fastField) {
          fastField->Advance(0, byteCount);
        } else {
          HandleRelativePosition(byteCount);
        }
      } else if (AdvanceRecord()) {
        if (fastField) {
          fastField->NextRecord(*this);
        }
      } else {
        return common::nullopt;
      }
      ch = GetCurrentChar(byteCount, fastField);
    }
    return ch;
  }

  template <Direction D>
  RT_API_ATTRS bool CheckFormattedStmtType(const char *name) {
    if (get_if<FormattedIoStatementState<D>>()) {
      return true;
    } else {
      auto &handler{GetIoErrorHandler()};
      if (!handler.InError()) {
        handler.Crash("%s called for I/O statement that is not formatted %s",
            name, D == Direction::Output ? "output" : "input");
      }
      return false;
    }
  }

private:
  std::variant<common::reference_wrapper<OpenStatementState>,
      common::reference_wrapper<CloseStatementState>,
      common::reference_wrapper<NoopStatementState>,
      common::reference_wrapper<
          InternalFormattedIoStatementState<Direction::Output>>,
      common::reference_wrapper<
          InternalFormattedIoStatementState<Direction::Input>>,
      common::reference_wrapper<
          InternalListIoStatementState<Direction::Output>>,
      common::reference_wrapper<InternalListIoStatementState<Direction::Input>>,
      common::reference_wrapper<
          ExternalFormattedIoStatementState<Direction::Output>>,
      common::reference_wrapper<
          ExternalFormattedIoStatementState<Direction::Input>>,
      common::reference_wrapper<
          ExternalListIoStatementState<Direction::Output>>,
      common::reference_wrapper<ExternalListIoStatementState<Direction::Input>>,
      common::reference_wrapper<
          ExternalUnformattedIoStatementState<Direction::Output>>,
      common::reference_wrapper<
          ExternalUnformattedIoStatementState<Direction::Input>>,
      common::reference_wrapper<
          ChildFormattedIoStatementState<Direction::Output>>,
      common::reference_wrapper<
          ChildFormattedIoStatementState<Direction::Input>>,
      common::reference_wrapper<ChildListIoStatementState<Direction::Output>>,
      common::reference_wrapper<ChildListIoStatementState<Direction::Input>>,
      common::reference_wrapper<
          ChildUnformattedIoStatementState<Direction::Output>>,
      common::reference_wrapper<
          ChildUnformattedIoStatementState<Direction::Input>>,
      common::reference_wrapper<InquireUnitState>,
      common::reference_wrapper<InquireNoUnitState>,
      common::reference_wrapper<InquireUnconnectedFileState>,
      common::reference_wrapper<InquireIOLengthState>,
      common::reference_wrapper<ExternalMiscIoStatementState>,
      common::reference_wrapper<ErroneousIoStatementState>>
      u_;
};

// Base class for all per-I/O statement state classes.
class IoStatementBase : public IoErrorHandler {
public:
  using IoErrorHandler::IoErrorHandler;

  RT_API_ATTRS bool completedOperation() const { return completedOperation_; }
  RT_API_ATTRS const NonTbpDefinedIoTable *nonTbpDefinedIoTable() const {
    return nonTbpDefinedIoTable_;
  }
  RT_API_ATTRS void set_nonTbpDefinedIoTable(
      const NonTbpDefinedIoTable *table) {
    nonTbpDefinedIoTable_ = table;
  }

  RT_API_ATTRS void CompleteOperation() { completedOperation_ = true; }
  RT_API_ATTRS int EndIoStatement() { return GetIoStat(); }

  // These are default no-op backstops that can be overridden by descendants.
  RT_API_ATTRS bool Emit(
      const char *, std::size_t bytes, std::size_t elementBytes = 0);
  RT_API_ATTRS bool Receive(
      char *, std::size_t bytes, std::size_t elementBytes = 0);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&);
  RT_API_ATTRS std::size_t ViewBytesInRecord(const char *&, bool forward) const;
  RT_API_ATTRS bool AdvanceRecord(int);
  RT_API_ATTRS void BackspaceRecord();
  RT_API_ATTRS void HandleRelativePosition(std::int64_t);
  RT_API_ATTRS void HandleAbsolutePosition(std::int64_t);
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1);
  RT_API_ATTRS ExternalFileUnit *GetExternalFileUnit() const;
  RT_API_ATTRS bool BeginReadingRecord();
  RT_API_ATTRS void FinishReadingRecord();
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, char *, std::size_t);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t &);
  RT_API_ATTRS std::int64_t InquirePos();

  RT_API_ATTRS void BadInquiryKeywordHashCrash(InquiryKeywordHash);

  RT_API_ATTRS void ReportUnsupportedChildIo() const {
    Crash("not yet implemented: child IO");
  }

protected:
  bool completedOperation_{false};

private:
  // Original NonTbpDefinedIoTable argument to Input/OutputDerivedType,
  // saved here so that it can also be used in child I/O statements.
  const NonTbpDefinedIoTable *nonTbpDefinedIoTable_{nullptr};
};

// Common state for list-directed & NAMELIST I/O, both internal & external
template <Direction> class ListDirectedStatementState;
template <>
class ListDirectedStatementState<Direction::Output>
    : public FormattedIoStatementState<Direction::Output> {
public:
  RT_API_ATTRS bool EmitLeadingSpaceOrAdvance(
      IoStatementState &, std::size_t = 1, bool isCharacter = false);
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1);
  RT_API_ATTRS bool lastWasUndelimitedCharacter() const {
    return lastWasUndelimitedCharacter_;
  }
  RT_API_ATTRS void set_lastWasUndelimitedCharacter(bool yes = true) {
    lastWasUndelimitedCharacter_ = yes;
  }

private:
  bool lastWasUndelimitedCharacter_{false};
};
template <>
class ListDirectedStatementState<Direction::Input>
    : public FormattedIoStatementState<Direction::Input> {
public:
  RT_API_ATTRS const NamelistGroup *namelistGroup() const {
    return namelistGroup_;
  }
  RT_API_ATTRS int EndIoStatement();

  // Skips value separators, handles repetition and null values.
  // Vacant when '/' appears; present with descriptor == ListDirectedNullValue
  // when a null value appears.
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1);

  // Each NAMELIST input item is treated like a distinct list-directed
  // input statement.  This member function resets some state so that
  // repetition and null values work correctly for each successive
  // NAMELIST input item.
  RT_API_ATTRS void ResetForNextNamelistItem(
      const NamelistGroup *namelistGroup) {
    remaining_ = 0;
    if (repeatPosition_) {
      repeatPosition_->Cancel();
    }
    eatComma_ = false;
    realPart_ = imaginaryPart_ = false;
    namelistGroup_ = namelistGroup;
  }

protected:
  const NamelistGroup *namelistGroup_{nullptr};

private:
  int remaining_{0}; // for "r*" repetition
  common::optional<SavedPosition> repeatPosition_;
  bool eatComma_{false}; // consume comma after previously read item
  bool hitSlash_{false}; // once '/' is seen, nullify further items
  bool realPart_{false};
  bool imaginaryPart_{false};
};

template <Direction DIR>
class InternalIoStatementState : public IoStatementBase,
                                 public IoDirectionState<DIR> {
public:
  using Buffer =
      std::conditional_t<DIR == Direction::Input, const char *, char *>;
  RT_API_ATTRS InternalIoStatementState(Buffer, std::size_t,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS InternalIoStatementState(
      const Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS int EndIoStatement();

  RT_API_ATTRS bool Emit(
      const char *data, std::size_t bytes, std::size_t elementBytes = 0);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&);
  RT_API_ATTRS bool AdvanceRecord(int = 1);
  RT_API_ATTRS void BackspaceRecord();
  RT_API_ATTRS ConnectionState &GetConnectionState() { return unit_; }
  RT_API_ATTRS MutableModes &mutableModes() { return unit_.modes; }
  RT_API_ATTRS void HandleRelativePosition(std::int64_t);
  RT_API_ATTRS void HandleAbsolutePosition(std::int64_t);
  RT_API_ATTRS std::int64_t InquirePos();

protected:
  bool free_{true};
  InternalDescriptorUnit<DIR> unit_;
};

template <Direction DIR, typename CHAR>
class InternalFormattedIoStatementState
    : public InternalIoStatementState<DIR>,
      public FormattedIoStatementState<DIR> {
public:
  using CharType = CHAR;
  using typename InternalIoStatementState<DIR>::Buffer;
  RT_API_ATTRS InternalFormattedIoStatementState(Buffer internal,
      std::size_t internalLength, const CharType *format,
      std::size_t formatLength, const Descriptor *formatDescriptor = nullptr,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS InternalFormattedIoStatementState(const Descriptor &,
      const CharType *format, std::size_t formatLength,
      const Descriptor *formatDescriptor = nullptr,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS IoStatementState &ioStatementState() {
    return ioStatementState_;
  }
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }

private:
  IoStatementState ioStatementState_; // points to *this
  using InternalIoStatementState<DIR>::unit_;
  // format_ *must* be last; it may be partial someday
  FormatControl<InternalFormattedIoStatementState> format_;
};

template <Direction DIR>
class InternalListIoStatementState : public InternalIoStatementState<DIR>,
                                     public ListDirectedStatementState<DIR> {
public:
  using typename InternalIoStatementState<DIR>::Buffer;
  RT_API_ATTRS InternalListIoStatementState(Buffer internal,
      std::size_t internalLength, const char *sourceFile = nullptr,
      int sourceLine = 0);
  RT_API_ATTRS InternalListIoStatementState(
      const Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS IoStatementState &ioStatementState() {
    return ioStatementState_;
  }
  using ListDirectedStatementState<DIR>::GetNextDataEdit;
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();

private:
  IoStatementState ioStatementState_; // points to *this
  using InternalIoStatementState<DIR>::unit_;
};

class ExternalIoStatementBase : public IoStatementBase {
public:
  RT_API_ATTRS ExternalIoStatementBase(
      ExternalFileUnit &, const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS ExternalFileUnit &unit() { return unit_; }
  RT_API_ATTRS const ExternalFileUnit &unit() const { return unit_; }
  RT_API_ATTRS MutableModes &mutableModes();
  RT_API_ATTRS ConnectionState &GetConnectionState();
  RT_API_ATTRS int asynchronousID() const { return asynchronousID_; }
  RT_API_ATTRS void set_destroy(bool yes = true) { destroy_ = yes; }
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS ExternalFileUnit *GetExternalFileUnit() const { return &unit_; }
  RT_API_ATTRS void SetAsynchronous();
  RT_API_ATTRS std::int64_t InquirePos();

private:
  ExternalFileUnit &unit_;
  int asynchronousID_{-1};
  bool destroy_{false};
};

template <Direction DIR>
class ExternalIoStatementState : public ExternalIoStatementBase,
                                 public IoDirectionState<DIR> {
public:
  RT_API_ATTRS ExternalIoStatementState(
      ExternalFileUnit &, const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS MutableModes &mutableModes() { return mutableModes_; }
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS bool Emit(
      const char *, std::size_t bytes, std::size_t elementBytes = 0);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&);
  RT_API_ATTRS std::size_t ViewBytesInRecord(const char *&, bool forward) const;
  RT_API_ATTRS bool AdvanceRecord(int = 1);
  RT_API_ATTRS void BackspaceRecord();
  RT_API_ATTRS void HandleRelativePosition(std::int64_t);
  RT_API_ATTRS void HandleAbsolutePosition(std::int64_t);
  RT_API_ATTRS bool BeginReadingRecord();
  RT_API_ATTRS void FinishReadingRecord();

private:
  // These are forked from ConnectionState's modes at the beginning
  // of each formatted I/O statement so they may be overridden by control
  // edit descriptors during the statement.
  MutableModes mutableModes_;
};

template <Direction DIR, typename CHAR>
class ExternalFormattedIoStatementState
    : public ExternalIoStatementState<DIR>,
      public FormattedIoStatementState<DIR> {
public:
  using CharType = CHAR;
  RT_API_ATTRS ExternalFormattedIoStatementState(ExternalFileUnit &,
      const CharType *format, std::size_t formatLength,
      const Descriptor *formatDescriptor = nullptr,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }

private:
  FormatControl<ExternalFormattedIoStatementState> format_;
};

template <Direction DIR>
class ExternalListIoStatementState : public ExternalIoStatementState<DIR>,
                                     public ListDirectedStatementState<DIR> {
public:
  using ExternalIoStatementState<DIR>::ExternalIoStatementState;
  using ListDirectedStatementState<DIR>::GetNextDataEdit;
  RT_API_ATTRS int EndIoStatement();
};

template <Direction DIR>
class ExternalUnformattedIoStatementState
    : public ExternalIoStatementState<DIR> {
public:
  using ExternalIoStatementState<DIR>::ExternalIoStatementState;
  RT_API_ATTRS bool Receive(char *, std::size_t, std::size_t elementBytes = 0);
};

template <Direction DIR>
class ChildIoStatementState : public IoStatementBase,
                              public IoDirectionState<DIR> {
public:
  RT_API_ATTRS ChildIoStatementState(
      ChildIo &, const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS const NonTbpDefinedIoTable *nonTbpDefinedIoTable() const;
  RT_API_ATTRS void set_nonTbpDefinedIoTable(const NonTbpDefinedIoTable *);
  RT_API_ATTRS ChildIo &child() { return child_; }
  RT_API_ATTRS MutableModes &mutableModes() { return mutableModes_; }
  RT_API_ATTRS ConnectionState &GetConnectionState();
  RT_API_ATTRS ExternalFileUnit *GetExternalFileUnit() const;
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS bool Emit(
      const char *, std::size_t bytes, std::size_t elementBytes = 0);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&);
  RT_API_ATTRS std::size_t ViewBytesInRecord(const char *&, bool forward) const;
  RT_API_ATTRS void HandleRelativePosition(std::int64_t);
  RT_API_ATTRS void HandleAbsolutePosition(std::int64_t);

private:
  ChildIo &child_;
  MutableModes mutableModes_;
};

template <Direction DIR, typename CHAR>
class ChildFormattedIoStatementState : public ChildIoStatementState<DIR>,
                                       public FormattedIoStatementState<DIR> {
public:
  using CharType = CHAR;
  RT_API_ATTRS ChildFormattedIoStatementState(ChildIo &, const CharType *format,
      std::size_t formatLength, const Descriptor *formatDescriptor = nullptr,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS bool AdvanceRecord(int = 1);
  RT_API_ATTRS common::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }

private:
  FormatControl<ChildFormattedIoStatementState> format_;
};

template <Direction DIR>
class ChildListIoStatementState : public ChildIoStatementState<DIR>,
                                  public ListDirectedStatementState<DIR> {
public:
  RT_API_ATTRS ChildListIoStatementState(
      ChildIo &, const char *sourceFile = nullptr, int sourceLine = 0);
  using ListDirectedStatementState<DIR>::GetNextDataEdit;
  RT_API_ATTRS bool AdvanceRecord(int = 1);
  RT_API_ATTRS int EndIoStatement();
};

template <Direction DIR>
class ChildUnformattedIoStatementState : public ChildIoStatementState<DIR> {
public:
  using ChildIoStatementState<DIR>::ChildIoStatementState;
  RT_API_ATTRS bool Receive(char *, std::size_t, std::size_t elementBytes = 0);
};

// OPEN
class OpenStatementState : public ExternalIoStatementBase {
public:
  RT_API_ATTRS OpenStatementState(ExternalFileUnit &unit, bool wasExtant,
      bool isNewUnit, const char *sourceFile = nullptr, int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine},
        wasExtant_{wasExtant}, isNewUnit_{isNewUnit} {}
  RT_API_ATTRS bool wasExtant() const { return wasExtant_; }
  RT_API_ATTRS void set_status(OpenStatus status) {
    status_ = status;
  } // STATUS=
  RT_API_ATTRS void set_path(const char *, std::size_t); // FILE=
  RT_API_ATTRS void set_position(Position position) {
    position_ = position;
  } // POSITION=
  RT_API_ATTRS void set_action(Action action) { action_ = action; } // ACTION=
  RT_API_ATTRS void set_convert(Convert convert) {
    convert_ = convert;
  } // CONVERT=
  RT_API_ATTRS void set_access(Access access) { access_ = access; } // ACCESS=
  RT_API_ATTRS void set_isUnformatted(bool yes = true) {
    isUnformatted_ = yes;
  } // FORM=
  RT_API_ATTRS void set_mustBeFormatted(bool yes = true) {
    mustBeFormatted_ = yes;
  }

  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();

private:
  bool wasExtant_;
  bool isNewUnit_;
  common::optional<OpenStatus> status_;
  common::optional<Position> position_;
  common::optional<Action> action_;
  Convert convert_{Convert::Unknown};
  OwningPtr<char> path_;
  std::size_t pathLength_{};
  common::optional<bool> isUnformatted_;
  common::optional<bool> mustBeFormatted_;
  common::optional<Access> access_;
};

class CloseStatementState : public ExternalIoStatementBase {
public:
  RT_API_ATTRS CloseStatementState(ExternalFileUnit &unit,
      const char *sourceFile = nullptr, int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine} {}
  RT_API_ATTRS void set_status(CloseStatus status) { status_ = status; }
  RT_API_ATTRS int EndIoStatement();

private:
  CloseStatus status_{CloseStatus::Keep};
};

// For CLOSE(bad unit), WAIT(bad unit, ID=nonzero), INQUIRE(unconnected unit),
// and recoverable BACKSPACE(bad unit)
class NoUnitIoStatementState : public IoStatementBase {
public:
  RT_API_ATTRS IoStatementState &ioStatementState() {
    return ioStatementState_;
  }
  RT_API_ATTRS MutableModes &mutableModes() { return connection_.modes; }
  RT_API_ATTRS ConnectionState &GetConnectionState() { return connection_; }
  RT_API_ATTRS int badUnitNumber() const { return badUnitNumber_; }
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();

protected:
  template <typename A>
  RT_API_ATTRS NoUnitIoStatementState(A &stmt, const char *sourceFile = nullptr,
      int sourceLine = 0, int badUnitNumber = -1)
      : IoStatementBase{sourceFile, sourceLine}, ioStatementState_{stmt},
        badUnitNumber_{badUnitNumber} {}

private:
  IoStatementState ioStatementState_; // points to *this
  ConnectionState connection_;
  int badUnitNumber_;
};

class NoopStatementState : public NoUnitIoStatementState {
public:
  RT_API_ATTRS NoopStatementState(
      const char *sourceFile = nullptr, int sourceLine = 0, int unitNumber = -1)
      : NoUnitIoStatementState{*this, sourceFile, sourceLine, unitNumber} {}
  RT_API_ATTRS void set_status(CloseStatus) {} // discards
};

extern template class InternalIoStatementState<Direction::Output>;
extern template class InternalIoStatementState<Direction::Input>;
extern template class InternalFormattedIoStatementState<Direction::Output>;
extern template class InternalFormattedIoStatementState<Direction::Input>;
extern template class InternalListIoStatementState<Direction::Output>;
extern template class InternalListIoStatementState<Direction::Input>;
extern template class ExternalIoStatementState<Direction::Output>;
extern template class ExternalIoStatementState<Direction::Input>;
extern template class ExternalFormattedIoStatementState<Direction::Output>;
extern template class ExternalFormattedIoStatementState<Direction::Input>;
extern template class ExternalListIoStatementState<Direction::Output>;
extern template class ExternalListIoStatementState<Direction::Input>;
extern template class ExternalUnformattedIoStatementState<Direction::Output>;
extern template class ExternalUnformattedIoStatementState<Direction::Input>;
extern template class ChildIoStatementState<Direction::Output>;
extern template class ChildIoStatementState<Direction::Input>;
extern template class ChildFormattedIoStatementState<Direction::Output>;
extern template class ChildFormattedIoStatementState<Direction::Input>;
extern template class ChildListIoStatementState<Direction::Output>;
extern template class ChildListIoStatementState<Direction::Input>;
extern template class ChildUnformattedIoStatementState<Direction::Output>;
extern template class ChildUnformattedIoStatementState<Direction::Input>;

extern template class FormatControl<
    InternalFormattedIoStatementState<Direction::Output>>;
extern template class FormatControl<
    InternalFormattedIoStatementState<Direction::Input>>;
extern template class FormatControl<
    ExternalFormattedIoStatementState<Direction::Output>>;
extern template class FormatControl<
    ExternalFormattedIoStatementState<Direction::Input>>;
extern template class FormatControl<
    ChildFormattedIoStatementState<Direction::Output>>;
extern template class FormatControl<
    ChildFormattedIoStatementState<Direction::Input>>;

class InquireUnitState : public ExternalIoStatementBase {
public:
  RT_API_ATTRS InquireUnitState(ExternalFileUnit &unit,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, char *, std::size_t);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t &);
};

class InquireNoUnitState : public NoUnitIoStatementState {
public:
  RT_API_ATTRS InquireNoUnitState(const char *sourceFile = nullptr,
      int sourceLine = 0, int badUnitNumber = -1);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, char *, std::size_t);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t &);
};

class InquireUnconnectedFileState : public NoUnitIoStatementState {
public:
  RT_API_ATTRS InquireUnconnectedFileState(OwningPtr<char> &&path,
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, char *, std::size_t);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  RT_API_ATTRS bool Inquire(InquiryKeywordHash, std::int64_t &);

private:
  OwningPtr<char> path_; // trimmed and NUL terminated
};

class InquireIOLengthState : public NoUnitIoStatementState,
                             public OutputStatementState {
public:
  RT_API_ATTRS InquireIOLengthState(
      const char *sourceFile = nullptr, int sourceLine = 0);
  RT_API_ATTRS std::size_t bytes() const { return bytes_; }
  RT_API_ATTRS bool Emit(
      const char *, std::size_t bytes, std::size_t elementBytes = 0);

private:
  std::size_t bytes_{0};
};

class ExternalMiscIoStatementState : public ExternalIoStatementBase {
public:
  enum Which { Flush, Backspace, Endfile, Rewind, Wait };
  RT_API_ATTRS ExternalMiscIoStatementState(ExternalFileUnit &unit, Which which,
      const char *sourceFile = nullptr, int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine}, which_{which} {}
  RT_API_ATTRS void CompleteOperation();
  RT_API_ATTRS int EndIoStatement();

private:
  Which which_;
};

class ErroneousIoStatementState : public IoStatementBase {
public:
  explicit RT_API_ATTRS ErroneousIoStatementState(Iostat iostat,
      ExternalFileUnit *unit = nullptr, const char *sourceFile = nullptr,
      int sourceLine = 0)
      : IoStatementBase{sourceFile, sourceLine}, unit_{unit} {
    SetPendingError(iostat);
  }
  RT_API_ATTRS int EndIoStatement();
  RT_API_ATTRS ConnectionState &GetConnectionState() { return connection_; }
  RT_API_ATTRS MutableModes &mutableModes() { return connection_.modes; }

private:
  ConnectionState connection_;
  ExternalFileUnit *unit_{nullptr};
};

RT_OFFLOAD_API_GROUP_END

} // namespace Fortran::runtime::io
#endif // FLANG_RT_RUNTIME_IO_STMT_H_
