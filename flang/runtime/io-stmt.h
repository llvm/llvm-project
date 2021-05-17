//===-- runtime/io-stmt.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Representations of the state of an I/O statement in progress

#ifndef FORTRAN_RUNTIME_IO_STMT_H_
#define FORTRAN_RUNTIME_IO_STMT_H_

#include "connection.h"
#include "descriptor.h"
#include "file.h"
#include "format.h"
#include "internal-unit.h"
#include "io-api.h"
#include "io-error.h"
#include <functional>
#include <type_traits>
#include <variant>

namespace Fortran::runtime::io {

class ExternalFileUnit;

class OpenStatementState;
class InquireUnitState;
class InquireNoUnitState;
class InquireUnconnectedFileState;
class InquireIOLengthState;
class ExternalMiscIoStatementState;
class CloseStatementState;
class NoopCloseStatementState;

template <Direction, typename CHAR = char>
class InternalFormattedIoStatementState;
template <Direction, typename CHAR = char> class InternalListIoStatementState;
template <Direction, typename CHAR = char>
class ExternalFormattedIoStatementState;
template <Direction> class ExternalListIoStatementState;
template <Direction> class UnformattedIoStatementState;

struct InputStatementState {};
struct OutputStatementState {};
template <Direction D>
using IoDirectionState = std::conditional_t<D == Direction::Input,
    InputStatementState, OutputStatementState>;
struct FormattedIoStatementState {};

// The Cookie type in the I/O API is a pointer (for C) to this class.
class IoStatementState {
public:
  template <typename A> explicit IoStatementState(A &x) : u_{x} {}

  // These member functions each project themselves into the active alternative.
  // They're used by per-data-item routines in the I/O API (e.g., OutputReal64)
  // to interact with the state of the I/O statement in progress.
  // This design avoids virtual member functions and function pointers,
  // which may not have good support in some runtime environments.
  std::optional<DataEdit> GetNextDataEdit(int = 1);
  bool Emit(const char *, std::size_t, std::size_t elementBytes = 0);
  std::optional<char32_t> GetCurrentChar(); // vacant after end of record
  bool AdvanceRecord(int = 1);
  void BackspaceRecord();
  void HandleRelativePosition(std::int64_t);
  int EndIoStatement();
  ConnectionState &GetConnectionState();
  IoErrorHandler &GetIoErrorHandler() const;
  ExternalFileUnit *GetExternalFileUnit() const; // null if internal unit
  MutableModes &mutableModes();
  bool BeginReadingRecord();
  void FinishReadingRecord();
  bool Inquire(InquiryKeywordHash, char *, std::size_t);
  bool Inquire(InquiryKeywordHash, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t, bool &); // PENDING=
  bool Inquire(InquiryKeywordHash, std::int64_t &);

  // N.B.: this also works with base classes
  template <typename A> A *get_if() const {
    return std::visit(
        [](auto &x) -> A * {
          if constexpr (std::is_convertible_v<decltype(x.get()), A &>) {
            return &x.get();
          }
          return nullptr;
        },
        u_);
  }

  bool EmitRepeated(char, std::size_t);
  bool EmitField(const char *, std::size_t length, std::size_t width);

  // For fixed-width fields, initialize the number of remaining characters.
  // Skip over leading blanks, then return the first non-blank character (if
  // any).
  std::optional<char32_t> PrepareInput(
      const DataEdit &edit, std::optional<int> &remaining);

  std::optional<char32_t> SkipSpaces(std::optional<int> &remaining);
  std::optional<char32_t> NextInField(std::optional<int> &remaining);
  // Skips spaces, advances records, and ignores NAMELIST comments
  std::optional<char32_t> GetNextNonBlank();

  template <Direction D> void CheckFormattedStmtType(const char *name) {
    if (!get_if<FormattedIoStatementState>() ||
        !get_if<IoDirectionState<D>>()) {
      GetIoErrorHandler().Crash(
          "%s called for I/O statement that is not formatted %s", name,
          D == Direction::Output ? "output" : "input");
    }
  }

private:
  std::variant<std::reference_wrapper<OpenStatementState>,
      std::reference_wrapper<CloseStatementState>,
      std::reference_wrapper<NoopCloseStatementState>,
      std::reference_wrapper<
          InternalFormattedIoStatementState<Direction::Output>>,
      std::reference_wrapper<
          InternalFormattedIoStatementState<Direction::Input>>,
      std::reference_wrapper<InternalListIoStatementState<Direction::Output>>,
      std::reference_wrapper<InternalListIoStatementState<Direction::Input>>,
      std::reference_wrapper<
          ExternalFormattedIoStatementState<Direction::Output>>,
      std::reference_wrapper<
          ExternalFormattedIoStatementState<Direction::Input>>,
      std::reference_wrapper<ExternalListIoStatementState<Direction::Output>>,
      std::reference_wrapper<ExternalListIoStatementState<Direction::Input>>,
      std::reference_wrapper<UnformattedIoStatementState<Direction::Output>>,
      std::reference_wrapper<UnformattedIoStatementState<Direction::Input>>,
      std::reference_wrapper<InquireUnitState>,
      std::reference_wrapper<InquireNoUnitState>,
      std::reference_wrapper<InquireUnconnectedFileState>,
      std::reference_wrapper<InquireIOLengthState>,
      std::reference_wrapper<ExternalMiscIoStatementState>>
      u_;
};

// Base class for all per-I/O statement state classes.
// Inherits IoErrorHandler from its base.
struct IoStatementBase : public DefaultFormatControlCallbacks {
  using DefaultFormatControlCallbacks::DefaultFormatControlCallbacks;
  int EndIoStatement();
  std::optional<DataEdit> GetNextDataEdit(IoStatementState &, int = 1);
  ExternalFileUnit *GetExternalFileUnit() const { return nullptr; }
  bool BeginReadingRecord() { return true; }
  void FinishReadingRecord() {}
  bool Inquire(InquiryKeywordHash, char *, std::size_t);
  bool Inquire(InquiryKeywordHash, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t &);
  void BadInquiryKeywordHashCrash(InquiryKeywordHash);
};

// Common state for list-directed & NAMELIST I/O, both internal & external
template <Direction> class ListDirectedStatementState;
template <>
class ListDirectedStatementState<Direction::Output>
    : public FormattedIoStatementState {
public:
  bool EmitLeadingSpaceOrAdvance(
      IoStatementState &, std::size_t = 1, bool isCharacter = false);
  std::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1);
  bool lastWasUndelimitedCharacter() const {
    return lastWasUndelimitedCharacter_;
  }
  void set_lastWasUndelimitedCharacter(bool yes = true) {
    lastWasUndelimitedCharacter_ = yes;
  }

private:
  bool lastWasUndelimitedCharacter_{false};
};
template <>
class ListDirectedStatementState<Direction::Input>
    : public FormattedIoStatementState {
public:
  // Skips value separators, handles repetition and null values.
  // Vacant when '/' appears; present with descriptor == ListDirectedNullValue
  // when a null value appears.
  std::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1);

private:
  int remaining_{0}; // for "r*" repetition
  std::int64_t initialRecordNumber_;
  std::int64_t initialPositionInRecord_;
  bool isFirstItem_{true}; // leading separator implies null first item
  bool hitSlash_{false}; // once '/' is seen, nullify further items
  bool realPart_{false};
  bool imaginaryPart_{false};
};

template <Direction DIR, typename CHAR = char>
class InternalIoStatementState : public IoStatementBase,
                                 public IoDirectionState<DIR> {
public:
  using CharType = CHAR;
  using Buffer =
      std::conditional_t<DIR == Direction::Input, const CharType *, CharType *>;
  InternalIoStatementState(Buffer, std::size_t,
      const char *sourceFile = nullptr, int sourceLine = 0);
  InternalIoStatementState(
      const Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
  int EndIoStatement();
  bool Emit(const CharType *, std::size_t chars /* not necessarily bytes */,
      std::size_t elementBytes = 0);
  std::optional<char32_t> GetCurrentChar();
  bool AdvanceRecord(int = 1);
  void BackspaceRecord();
  ConnectionState &GetConnectionState() { return unit_; }
  MutableModes &mutableModes() { return unit_.modes; }
  void HandleRelativePosition(std::int64_t);
  void HandleAbsolutePosition(std::int64_t);

protected:
  bool free_{true};
  InternalDescriptorUnit<DIR> unit_;
};

template <Direction DIR, typename CHAR>
class InternalFormattedIoStatementState
    : public InternalIoStatementState<DIR, CHAR>,
      public FormattedIoStatementState {
public:
  using CharType = CHAR;
  using typename InternalIoStatementState<DIR, CharType>::Buffer;
  InternalFormattedIoStatementState(Buffer internal, std::size_t internalLength,
      const CharType *format, std::size_t formatLength,
      const char *sourceFile = nullptr, int sourceLine = 0);
  InternalFormattedIoStatementState(const Descriptor &, const CharType *format,
      std::size_t formatLength, const char *sourceFile = nullptr,
      int sourceLine = 0);
  IoStatementState &ioStatementState() { return ioStatementState_; }
  int EndIoStatement();
  std::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }

private:
  IoStatementState ioStatementState_; // points to *this
  using InternalIoStatementState<DIR, CharType>::unit_;
  // format_ *must* be last; it may be partial someday
  FormatControl<InternalFormattedIoStatementState> format_;
};

template <Direction DIR, typename CHAR>
class InternalListIoStatementState : public InternalIoStatementState<DIR, CHAR>,
                                     public ListDirectedStatementState<DIR> {
public:
  using CharType = CHAR;
  using typename InternalIoStatementState<DIR, CharType>::Buffer;
  InternalListIoStatementState(Buffer internal, std::size_t internalLength,
      const char *sourceFile = nullptr, int sourceLine = 0);
  InternalListIoStatementState(
      const Descriptor &, const char *sourceFile = nullptr, int sourceLine = 0);
  IoStatementState &ioStatementState() { return ioStatementState_; }
  using ListDirectedStatementState<DIR>::GetNextDataEdit;

private:
  IoStatementState ioStatementState_; // points to *this
  using InternalIoStatementState<DIR, CharType>::unit_;
};

class ExternalIoStatementBase : public IoStatementBase {
public:
  ExternalIoStatementBase(
      ExternalFileUnit &, const char *sourceFile = nullptr, int sourceLine = 0);
  ExternalFileUnit &unit() { return unit_; }
  MutableModes &mutableModes();
  ConnectionState &GetConnectionState();
  int EndIoStatement();
  ExternalFileUnit *GetExternalFileUnit() { return &unit_; }

private:
  ExternalFileUnit &unit_;
};

template <Direction DIR>
class ExternalIoStatementState : public ExternalIoStatementBase,
                                 public IoDirectionState<DIR> {
public:
  using ExternalIoStatementBase::ExternalIoStatementBase;
  int EndIoStatement();
  bool Emit(const char *, std::size_t, std::size_t elementBytes = 0);
  bool Emit(const char16_t *, std::size_t chars /* not bytes */);
  bool Emit(const char32_t *, std::size_t chars /* not bytes */);
  std::optional<char32_t> GetCurrentChar();
  bool AdvanceRecord(int = 1);
  void BackspaceRecord();
  void HandleRelativePosition(std::int64_t);
  void HandleAbsolutePosition(std::int64_t);
  bool BeginReadingRecord();
  void FinishReadingRecord();
};

template <Direction DIR, typename CHAR>
class ExternalFormattedIoStatementState : public ExternalIoStatementState<DIR>,
                                          public FormattedIoStatementState {
public:
  using CharType = CHAR;
  ExternalFormattedIoStatementState(ExternalFileUnit &, const CharType *format,
      std::size_t formatLength, const char *sourceFile = nullptr,
      int sourceLine = 0);
  MutableModes &mutableModes() { return mutableModes_; }
  int EndIoStatement();
  std::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1) {
    return format_.GetNextDataEdit(*this, maxRepeat);
  }

private:
  // These are forked from ConnectionState's modes at the beginning
  // of each formatted I/O statement so they may be overridden by control
  // edit descriptors during the statement.
  MutableModes mutableModes_;
  FormatControl<ExternalFormattedIoStatementState> format_;
};

template <Direction DIR>
class ExternalListIoStatementState : public ExternalIoStatementState<DIR>,
                                     public ListDirectedStatementState<DIR> {
public:
  using ExternalIoStatementState<DIR>::ExternalIoStatementState;
  using ListDirectedStatementState<DIR>::GetNextDataEdit;
};

template <Direction DIR>
class UnformattedIoStatementState : public ExternalIoStatementState<DIR> {
public:
  using ExternalIoStatementState<DIR>::ExternalIoStatementState;
  bool Receive(char *, std::size_t, std::size_t elementBytes = 0);
  bool Emit(const char *, std::size_t, std::size_t elementBytes = 0);
};

class OpenStatementState : public ExternalIoStatementBase {
public:
  OpenStatementState(ExternalFileUnit &unit, bool wasExtant,
      const char *sourceFile = nullptr, int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine}, wasExtant_{
                                                                   wasExtant} {}
  bool wasExtant() const { return wasExtant_; }
  void set_status(OpenStatus status) { status_ = status; } // STATUS=
  void set_path(const char *, std::size_t); // FILE=
  void set_position(Position position) { position_ = position; } // POSITION=
  void set_action(Action action) { action_ = action; } // ACTION=
  void set_convert(Convert convert) { convert_ = convert; } // CONVERT=
  void set_access(Access access) { access_ = access; } // ACCESS=
  void set_isUnformatted(bool yes = true) { isUnformatted_ = yes; } // FORM=
  int EndIoStatement();

private:
  bool wasExtant_;
  std::optional<OpenStatus> status_;
  Position position_{Position::AsIs};
  std::optional<Action> action_;
  Convert convert_{Convert::Native};
  OwningPtr<char> path_;
  std::size_t pathLength_;
  std::optional<bool> isUnformatted_;
  std::optional<Access> access_;
};

class CloseStatementState : public ExternalIoStatementBase {
public:
  CloseStatementState(ExternalFileUnit &unit, const char *sourceFile = nullptr,
      int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine} {}
  void set_status(CloseStatus status) { status_ = status; }
  int EndIoStatement();

private:
  CloseStatus status_{CloseStatus::Keep};
};

// For CLOSE(bad unit) and INQUIRE(unconnected unit)
class NoUnitIoStatementState : public IoStatementBase {
public:
  IoStatementState &ioStatementState() { return ioStatementState_; }
  MutableModes &mutableModes() { return connection_.modes; }
  ConnectionState &GetConnectionState() { return connection_; }
  int EndIoStatement();

protected:
  template <typename A>
  NoUnitIoStatementState(const char *sourceFile, int sourceLine, A &stmt)
      : IoStatementBase{sourceFile, sourceLine}, ioStatementState_{stmt} {}

private:
  IoStatementState ioStatementState_; // points to *this
  ConnectionState connection_;
};

class NoopCloseStatementState : public NoUnitIoStatementState {
public:
  NoopCloseStatementState(const char *sourceFile, int sourceLine)
      : NoUnitIoStatementState{sourceFile, sourceLine, *this} {}
  void set_status(CloseStatus) {} // discards
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
extern template class UnformattedIoStatementState<Direction::Output>;
extern template class UnformattedIoStatementState<Direction::Input>;
extern template class FormatControl<
    InternalFormattedIoStatementState<Direction::Output>>;
extern template class FormatControl<
    InternalFormattedIoStatementState<Direction::Input>>;
extern template class FormatControl<
    ExternalFormattedIoStatementState<Direction::Output>>;
extern template class FormatControl<
    ExternalFormattedIoStatementState<Direction::Input>>;

class InquireUnitState : public ExternalIoStatementBase {
public:
  InquireUnitState(ExternalFileUnit &unit, const char *sourceFile = nullptr,
      int sourceLine = 0);
  bool Inquire(InquiryKeywordHash, char *, std::size_t);
  bool Inquire(InquiryKeywordHash, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t &);
};

class InquireNoUnitState : public NoUnitIoStatementState {
public:
  InquireNoUnitState(const char *sourceFile = nullptr, int sourceLine = 0);
  bool Inquire(InquiryKeywordHash, char *, std::size_t);
  bool Inquire(InquiryKeywordHash, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t &);
};

class InquireUnconnectedFileState : public NoUnitIoStatementState {
public:
  InquireUnconnectedFileState(OwningPtr<char> &&path,
      const char *sourceFile = nullptr, int sourceLine = 0);
  bool Inquire(InquiryKeywordHash, char *, std::size_t);
  bool Inquire(InquiryKeywordHash, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t, bool &);
  bool Inquire(InquiryKeywordHash, std::int64_t &);

private:
  OwningPtr<char> path_; // trimmed and NUL terminated
};

class InquireIOLengthState : public NoUnitIoStatementState,
                             public OutputStatementState {
public:
  InquireIOLengthState(const char *sourceFile = nullptr, int sourceLine = 0);
  std::size_t bytes() const { return bytes_; }
  bool Emit(const char *, std::size_t, std::size_t elementBytes = 0);

private:
  std::size_t bytes_{0};
};

class ExternalMiscIoStatementState : public ExternalIoStatementBase {
public:
  enum Which { Flush, Backspace, Endfile, Rewind };
  ExternalMiscIoStatementState(ExternalFileUnit &unit, Which which,
      const char *sourceFile = nullptr, int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine}, which_{which} {}
  int EndIoStatement();

private:
  Which which_;
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_STMT_H_
