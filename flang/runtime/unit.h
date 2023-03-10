//===-- runtime/unit.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fortran external I/O units

#ifndef FORTRAN_RUNTIME_IO_UNIT_H_
#define FORTRAN_RUNTIME_IO_UNIT_H_

#include "buffer.h"
#include "connection.h"
#include "environment.h"
#include "file.h"
#include "format.h"
#include "io-error.h"
#include "io-stmt.h"
#include "lock.h"
#include "terminator.h"
#include "flang/Common/constexpr-bitset.h"
#include "flang/Runtime/memory.h"
#include <cstdlib>
#include <cstring>
#include <optional>
#include <variant>

namespace Fortran::runtime::io {

class UnitMap;
class ChildIo;

class ExternalFileUnit : public ConnectionState,
                         public OpenFile,
                         public FileFrame<ExternalFileUnit> {
public:
  explicit ExternalFileUnit(int unitNumber) : unitNumber_{unitNumber} {
    isUTF8 = executionEnvironment.defaultUTF8;
    asyncIdAvailable_.set();
    asyncIdAvailable_.reset(0);
  }
  ~ExternalFileUnit() {}

  int unitNumber() const { return unitNumber_; }
  bool swapEndianness() const { return swapEndianness_; }
  bool createdForInternalChildIo() const { return createdForInternalChildIo_; }

  static ExternalFileUnit *LookUp(int unit);
  static ExternalFileUnit *LookUpOrCreate(
      int unit, const Terminator &, bool &wasExtant);
  static ExternalFileUnit *LookUpOrCreateAnonymous(int unit, Direction,
      std::optional<bool> isUnformatted, const Terminator &);
  static ExternalFileUnit *LookUp(const char *path, std::size_t pathLen);
  static ExternalFileUnit &CreateNew(int unit, const Terminator &);
  static ExternalFileUnit *LookUpForClose(int unit);
  static ExternalFileUnit &NewUnit(const Terminator &, bool forChildIo);
  static void CloseAll(IoErrorHandler &);
  static void FlushAll(IoErrorHandler &);

  void OpenUnit(std::optional<OpenStatus>, std::optional<Action>, Position,
      OwningPtr<char> &&path, std::size_t pathLength, Convert,
      IoErrorHandler &);
  void OpenAnonymousUnit(std::optional<OpenStatus>, std::optional<Action>,
      Position, Convert, IoErrorHandler &);
  void CloseUnit(CloseStatus, IoErrorHandler &);
  void DestroyClosed();

  Iostat SetDirection(Direction);

  template <typename A, typename... X>
  IoStatementState &BeginIoStatement(const Terminator &terminator, X &&...xs) {
    // Take lock_ and hold it until EndIoStatement().
#if USE_PTHREADS
    if (!lock_.TakeIfNoDeadlock()) {
      terminator.Crash("Recursive I/O attempted on unit %d", unitNumber_);
    }
#else
    lock_.Take();
#endif
    A &state{u_.emplace<A>(std::forward<X>(xs)...)};
    if constexpr (!std::is_same_v<A, OpenStatementState>) {
      state.mutableModes() = ConnectionState::modes;
    }
    directAccessRecWasSet_ = false;
    io_.emplace(state);
    return *io_;
  }

  bool Emit(
      const char *, std::size_t, std::size_t elementBytes, IoErrorHandler &);
  bool Receive(char *, std::size_t, std::size_t elementBytes, IoErrorHandler &);
  std::size_t GetNextInputBytes(const char *&, IoErrorHandler &);
  bool BeginReadingRecord(IoErrorHandler &);
  void FinishReadingRecord(IoErrorHandler &);
  bool AdvanceRecord(IoErrorHandler &);
  void BackspaceRecord(IoErrorHandler &);
  void FlushOutput(IoErrorHandler &);
  void FlushIfTerminal(IoErrorHandler &);
  void Endfile(IoErrorHandler &);
  void Rewind(IoErrorHandler &);
  void EndIoStatement();
  bool SetStreamPos(std::int64_t, IoErrorHandler &); // one-based, for POS=
  bool SetDirectRec(std::int64_t, IoErrorHandler &); // one-based, for REC=
  std::int64_t InquirePos() const {
    // 12.6.2.11 defines POS=1 as the beginning of file
    return frameOffsetInFile_ + recordOffsetInFrame_ + positionInRecord + 1;
  }

  ChildIo *GetChildIo() { return child_.get(); }
  ChildIo &PushChildIo(IoStatementState &);
  void PopChildIo(ChildIo &);

  int GetAsynchronousId(IoErrorHandler &);
  bool Wait(int);

private:
  static UnitMap &CreateUnitMap();
  static UnitMap &GetUnitMap();
  const char *FrameNextInput(IoErrorHandler &, std::size_t);
  void SetPosition(std::int64_t, IoErrorHandler &); // zero-based
  void BeginSequentialVariableUnformattedInputRecord(IoErrorHandler &);
  void BeginVariableFormattedInputRecord(IoErrorHandler &);
  void BackspaceFixedRecord(IoErrorHandler &);
  void BackspaceVariableUnformattedRecord(IoErrorHandler &);
  void BackspaceVariableFormattedRecord(IoErrorHandler &);
  bool SetVariableFormattedRecordLength();
  void DoImpliedEndfile(IoErrorHandler &);
  void DoEndfile(IoErrorHandler &);
  void CommitWrites();
  bool CheckDirectAccess(IoErrorHandler &);
  void HitEndOnRead(IoErrorHandler &);
  std::int32_t ReadHeaderOrFooter(std::int64_t frameOffset);

  Lock lock_;

  int unitNumber_{-1};
  Direction direction_{Direction::Output};
  bool impliedEndfile_{false}; // sequential/stream output has taken place
  bool beganReadingRecord_{false};
  bool directAccessRecWasSet_{false}; // REC= appeared
  // Subtle: The beginning of the frame can't be allowed to advance
  // during a single list-directed READ due to the possibility of a
  // multi-record CHARACTER value with a "r*" repeat count.  So we
  // manage the frame and the current record therein separately.
  std::int64_t frameOffsetInFile_{0};
  std::size_t recordOffsetInFrame_{0}; // of currentRecordNumber
  bool swapEndianness_{false};
  bool createdForInternalChildIo_{false};
  common::BitSet<64> asyncIdAvailable_;

  // When a synchronous I/O statement is in progress on this unit, holds its
  // state.
  std::variant<std::monostate, OpenStatementState, CloseStatementState,
      ExternalFormattedIoStatementState<Direction::Output>,
      ExternalFormattedIoStatementState<Direction::Input>,
      ExternalListIoStatementState<Direction::Output>,
      ExternalListIoStatementState<Direction::Input>,
      ExternalUnformattedIoStatementState<Direction::Output>,
      ExternalUnformattedIoStatementState<Direction::Input>, InquireUnitState,
      ExternalMiscIoStatementState, ErroneousIoStatementState>
      u_;

  // Points to the active alternative (if any) in u_ for use as a Cookie
  std::optional<IoStatementState> io_;

  // A stack of child I/O pseudo-units for user-defined derived type
  // I/O that have this unit number.
  OwningPtr<ChildIo> child_;
};

// A pseudo-unit for child I/O statements in user-defined derived type
// I/O subroutines; it forwards operations to the parent I/O statement,
// which can also be a child I/O statement.
class ChildIo {
public:
  ChildIo(IoStatementState &parent, OwningPtr<ChildIo> &&previous)
      : parent_{parent}, previous_{std::move(previous)} {}

  IoStatementState &parent() const { return parent_; }

  void EndIoStatement();

  template <typename A, typename... X>
  IoStatementState &BeginIoStatement(X &&...xs) {
    A &state{u_.emplace<A>(std::forward<X>(xs)...)};
    io_.emplace(state);
    return *io_;
  }

  OwningPtr<ChildIo> AcquirePrevious() { return std::move(previous_); }

  Iostat CheckFormattingAndDirection(bool unformatted, Direction);

private:
  IoStatementState &parent_;
  OwningPtr<ChildIo> previous_;
  std::variant<std::monostate,
      ChildFormattedIoStatementState<Direction::Output>,
      ChildFormattedIoStatementState<Direction::Input>,
      ChildListIoStatementState<Direction::Output>,
      ChildListIoStatementState<Direction::Input>,
      ChildUnformattedIoStatementState<Direction::Output>,
      ChildUnformattedIoStatementState<Direction::Input>, InquireUnitState,
      ErroneousIoStatementState, ExternalMiscIoStatementState>
      u_;
  std::optional<IoStatementState> io_;
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_UNIT_H_
