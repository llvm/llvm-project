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
#include "io-error.h"
#include <functional>
#include <type_traits>
#include <variant>

namespace Fortran::runtime::io {

class ExternalFileUnit;

class OpenStatementState;
class CloseStatementState;
class NoopCloseStatementState;

template <Direction, typename CHAR = char>
class InternalFormattedIoStatementState;
template <Direction, typename CHAR = char> class InternalListIoStatementState;
template <Direction, typename CHAR = char>
class ExternalFormattedIoStatementState;
template <Direction> class ExternalListIoStatementState;
template <Direction> class UnformattedIoStatementState;

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
  bool Emit(const char *, std::size_t);
  std::optional<char32_t> GetCurrentChar(); // vacant after end of record
  bool AdvanceRecord(int = 1);
  void BackspaceRecord();
  void HandleRelativePosition(std::int64_t);
  int EndIoStatement();
  ConnectionState &GetConnectionState();
  IoErrorHandler &GetIoErrorHandler() const;
  ExternalFileUnit *GetExternalFileUnit() const; // null if internal unit
  MutableModes &mutableModes();

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
  void SkipSpaces(std::optional<int> &remaining);
  std::optional<char32_t> NextInField(std::optional<int> &remaining);
  std::optional<char32_t> GetNextNonBlank(); // can advance record

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
      std::reference_wrapper<UnformattedIoStatementState<Direction::Input>>>
      u_;
};

// Base class for all per-I/O statement state classes.
// Inherits IoErrorHandler from its base.
struct IoStatementBase : public DefaultFormatControlCallbacks {
  using DefaultFormatControlCallbacks::DefaultFormatControlCallbacks;
  int EndIoStatement();
  std::optional<DataEdit> GetNextDataEdit(IoStatementState &, int = 1);
  ExternalFileUnit *GetExternalFileUnit() const { return nullptr; }
};

struct InputStatementState {};
struct OutputStatementState {};
template <Direction D>
using IoDirectionState = std::conditional_t<D == Direction::Input,
    InputStatementState, OutputStatementState>;

struct FormattedStatementState {};

// Common state for list-directed internal & external I/O
template <Direction> struct ListDirectedStatementState {};
template <> struct ListDirectedStatementState<Direction::Output> {
  static std::size_t RemainingSpaceInRecord(const ConnectionState &);
  bool NeedAdvance(const ConnectionState &, std::size_t) const;
  bool EmitLeadingSpaceOrAdvance(
      IoStatementState &, std::size_t, bool isCharacter = false);
  std::optional<DataEdit> GetNextDataEdit(
      IoStatementState &, int maxRepeat = 1);
  bool lastWasUndelimitedCharacter{false};
};
template <> class ListDirectedStatementState<Direction::Input> {
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
  bool Emit(const CharType *, std::size_t chars /* not bytes */);
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
      public FormattedStatementState {
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
  bool Emit(const char *, std::size_t chars /* not bytes */);
  bool Emit(const char16_t *, std::size_t chars /* not bytes */);
  bool Emit(const char32_t *, std::size_t chars /* not bytes */);
  std::optional<char32_t> GetCurrentChar();
  bool AdvanceRecord(int = 1);
  void BackspaceRecord();
  void HandleRelativePosition(std::int64_t);
  void HandleAbsolutePosition(std::int64_t);
};

template <Direction DIR, typename CHAR>
class ExternalFormattedIoStatementState : public ExternalIoStatementState<DIR>,
                                          public FormattedStatementState {
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
  int EndIoStatement();
};

class OpenStatementState : public ExternalIoStatementBase {
public:
  OpenStatementState(ExternalFileUnit &unit, bool wasExtant,
      const char *sourceFile = nullptr, int sourceLine = 0)
      : ExternalIoStatementBase{unit, sourceFile, sourceLine}, wasExtant_{
                                                                   wasExtant} {}
  bool wasExtant() const { return wasExtant_; }
  void set_status(OpenStatus status) { status_ = status; }
  void set_path(const char *, std::size_t, int kind); // FILE=
  void set_position(Position position) { position_ = position; } // POSITION=
  int EndIoStatement();

private:
  bool wasExtant_;
  OpenStatus status_{OpenStatus::Unknown};
  Position position_{Position::AsIs};
  OwningPtr<char> path_;
  std::size_t pathLength_;
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

class NoopCloseStatementState : public IoStatementBase {
public:
  NoopCloseStatementState(const char *sourceFile, int sourceLine)
      : IoStatementBase{sourceFile, sourceLine}, ioStatementState_{*this} {}
  IoStatementState &ioStatementState() { return ioStatementState_; }
  void set_status(CloseStatus) {} // discards
  MutableModes &mutableModes() { return connection_.modes; }
  ConnectionState &GetConnectionState() { return connection_; }
  int EndIoStatement();

private:
  IoStatementState ioStatementState_; // points to *this
  ConnectionState connection_;
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

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_STMT_H_
