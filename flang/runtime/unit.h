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
#include "flang/Common/optional.h"
#include "flang/Runtime/memory.h"
#include <cstdlib>
#include <cstring>
#include <variant>

namespace Fortran::runtime::io {

class UnitMap;
class ChildIo;
class ExternalFileUnit;

RT_OFFLOAD_VAR_GROUP_BEGIN
// Predefined file units.
extern RT_VAR_ATTRS ExternalFileUnit *defaultInput; // unit 5
extern RT_VAR_ATTRS ExternalFileUnit *defaultOutput; // unit 6
extern RT_VAR_ATTRS ExternalFileUnit *errorOutput; // unit 0 extension
RT_OFFLOAD_VAR_GROUP_END

#if defined(RT_USE_PSEUDO_FILE_UNIT)
// A flavor of OpenFile class that pretends to be a terminal,
// and only provides basic buffering of the output
// in an internal buffer, and Write's the output
// using std::printf(). Since it does not rely on file system
// APIs, it can be used to implement external output
// for offload devices.
class PseudoOpenFile {
public:
  using FileOffset = std::int64_t;

  RT_API_ATTRS const char *path() const { return nullptr; }
  RT_API_ATTRS std::size_t pathLength() const { return 0; }
  RT_API_ATTRS void set_path(OwningPtr<char> &&, std::size_t bytes) {}
  RT_API_ATTRS bool mayRead() const { return false; }
  RT_API_ATTRS bool mayWrite() const { return true; }
  RT_API_ATTRS bool mayPosition() const { return false; }
  RT_API_ATTRS bool mayAsynchronous() const { return false; }
  RT_API_ATTRS void set_mayAsynchronous(bool yes);
  // Pretend to be a terminal to force the output
  // at the end of IO statement.
  RT_API_ATTRS bool isTerminal() const { return true; }
  RT_API_ATTRS bool isWindowsTextFile() const { return false; }
  RT_API_ATTRS Fortran::common::optional<FileOffset> knownSize() const;
  RT_API_ATTRS bool IsConnected() const { return false; }
  RT_API_ATTRS void Open(OpenStatus, Fortran::common::optional<Action>,
      Position, IoErrorHandler &);
  RT_API_ATTRS void Predefine(int fd) {}
  RT_API_ATTRS void Close(CloseStatus, IoErrorHandler &);
  RT_API_ATTRS std::size_t Read(FileOffset, char *, std::size_t minBytes,
      std::size_t maxBytes, IoErrorHandler &);
  RT_API_ATTRS std::size_t Write(
      FileOffset, const char *, std::size_t, IoErrorHandler &);
  RT_API_ATTRS void Truncate(FileOffset, IoErrorHandler &);
  RT_API_ATTRS int ReadAsynchronously(
      FileOffset, char *, std::size_t, IoErrorHandler &);
  RT_API_ATTRS int WriteAsynchronously(
      FileOffset, const char *, std::size_t, IoErrorHandler &);
  RT_API_ATTRS void Wait(int id, IoErrorHandler &);
  RT_API_ATTRS void WaitAll(IoErrorHandler &);
  RT_API_ATTRS Position InquirePosition() const;
};
#endif // defined(RT_USE_PSEUDO_FILE_UNIT)

#if !defined(RT_USE_PSEUDO_FILE_UNIT)
using OpenFileClass = OpenFile;
using FileFrameClass = FileFrame<ExternalFileUnit>;
#else // defined(RT_USE_PSEUDO_FILE_UNIT)
using OpenFileClass = PseudoOpenFile;
// Use not so big buffer for the pseudo file unit frame.
using FileFrameClass = FileFrame<ExternalFileUnit, 1024>;
#endif // defined(RT_USE_PSEUDO_FILE_UNIT)

class ExternalFileUnit : public ConnectionState,
                         public OpenFileClass,
                         public FileFrameClass {
public:
  static constexpr int maxAsyncIds{64 * 16};

  explicit RT_API_ATTRS ExternalFileUnit(int unitNumber)
      : unitNumber_{unitNumber} {
    isUTF8 = executionEnvironment.defaultUTF8;
    for (int j{0}; 64 * j < maxAsyncIds; ++j) {
      asyncIdAvailable_[j].set();
    }
    asyncIdAvailable_[0].reset(0);
  }
  RT_API_ATTRS ~ExternalFileUnit() {}

  RT_API_ATTRS int unitNumber() const { return unitNumber_; }
  RT_API_ATTRS bool swapEndianness() const { return swapEndianness_; }
  RT_API_ATTRS bool createdForInternalChildIo() const {
    return createdForInternalChildIo_;
  }

  static RT_API_ATTRS ExternalFileUnit *LookUp(int unit);
  static RT_API_ATTRS ExternalFileUnit *LookUpOrCreate(
      int unit, const Terminator &, bool &wasExtant);
  static RT_API_ATTRS ExternalFileUnit *LookUpOrCreateAnonymous(int unit,
      Direction, Fortran::common::optional<bool> isUnformatted,
      const Terminator &);
  static RT_API_ATTRS ExternalFileUnit *LookUp(
      const char *path, std::size_t pathLen);
  static RT_API_ATTRS ExternalFileUnit &CreateNew(int unit, const Terminator &);
  static RT_API_ATTRS ExternalFileUnit *LookUpForClose(int unit);
  static RT_API_ATTRS ExternalFileUnit &NewUnit(
      const Terminator &, bool forChildIo);
  static RT_API_ATTRS void CloseAll(IoErrorHandler &);
  static RT_API_ATTRS void FlushAll(IoErrorHandler &);

  // Returns true if an existing unit was closed
  RT_API_ATTRS bool OpenUnit(Fortran::common::optional<OpenStatus>,
      Fortran::common::optional<Action>, Position, OwningPtr<char> &&path,
      std::size_t pathLength, Convert, IoErrorHandler &);
  RT_API_ATTRS void OpenAnonymousUnit(Fortran::common::optional<OpenStatus>,
      Fortran::common::optional<Action>, Position, Convert, IoErrorHandler &);
  RT_API_ATTRS void CloseUnit(CloseStatus, IoErrorHandler &);
  RT_API_ATTRS void DestroyClosed();

  RT_API_ATTRS Iostat SetDirection(Direction);

  template <typename A, typename... X>
  RT_API_ATTRS IoStatementState &BeginIoStatement(
      const Terminator &terminator, X &&...xs) {
    // Take lock_ and hold it until EndIoStatement().
#if USE_PTHREADS
    if (!lock_.TakeIfNoDeadlock()) {
      terminator.Crash("Recursive I/O attempted on unit %d", unitNumber_);
    }
#else
    lock_.Take();
#endif
    RT_DIAG_PUSH
    RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN
    A &state{u_.emplace<A>(std::forward<X>(xs)...)};
    RT_DIAG_POP
    if constexpr (!std::is_same_v<A, OpenStatementState>) {
      state.mutableModes() = ConnectionState::modes;
    }
    directAccessRecWasSet_ = false;
    io_.emplace(state);
    return *io_;
  }

  RT_API_ATTRS bool Emit(
      const char *, std::size_t, std::size_t elementBytes, IoErrorHandler &);
  RT_API_ATTRS bool Receive(
      char *, std::size_t, std::size_t elementBytes, IoErrorHandler &);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&, IoErrorHandler &);
  RT_API_ATTRS bool BeginReadingRecord(IoErrorHandler &);
  RT_API_ATTRS void FinishReadingRecord(IoErrorHandler &);
  RT_API_ATTRS bool AdvanceRecord(IoErrorHandler &);
  RT_API_ATTRS void BackspaceRecord(IoErrorHandler &);
  RT_API_ATTRS void FlushOutput(IoErrorHandler &);
  RT_API_ATTRS void FlushIfTerminal(IoErrorHandler &);
  RT_API_ATTRS void Endfile(IoErrorHandler &);
  RT_API_ATTRS void Rewind(IoErrorHandler &);
  RT_API_ATTRS void EndIoStatement();
  RT_API_ATTRS bool SetStreamPos(
      std::int64_t, IoErrorHandler &); // one-based, for POS=
  RT_API_ATTRS bool SetDirectRec(
      std::int64_t, IoErrorHandler &); // one-based, for REC=
  RT_API_ATTRS std::int64_t InquirePos() const {
    // 12.6.2.11 defines POS=1 as the beginning of file
    return frameOffsetInFile_ + recordOffsetInFrame_ + positionInRecord + 1;
  }

  RT_API_ATTRS ChildIo *GetChildIo() { return child_.get(); }
  RT_API_ATTRS ChildIo &PushChildIo(IoStatementState &);
  RT_API_ATTRS void PopChildIo(ChildIo &);

  RT_API_ATTRS int GetAsynchronousId(IoErrorHandler &);
  RT_API_ATTRS bool Wait(int);

private:
  static RT_API_ATTRS UnitMap &CreateUnitMap();
  static RT_API_ATTRS UnitMap &GetUnitMap();
  RT_API_ATTRS const char *FrameNextInput(IoErrorHandler &, std::size_t);
  RT_API_ATTRS void SetPosition(std::int64_t, IoErrorHandler &); // zero-based
  RT_API_ATTRS void BeginSequentialVariableUnformattedInputRecord(
      IoErrorHandler &);
  RT_API_ATTRS void BeginVariableFormattedInputRecord(IoErrorHandler &);
  RT_API_ATTRS void BackspaceFixedRecord(IoErrorHandler &);
  RT_API_ATTRS void BackspaceVariableUnformattedRecord(IoErrorHandler &);
  RT_API_ATTRS void BackspaceVariableFormattedRecord(IoErrorHandler &);
  RT_API_ATTRS bool SetVariableFormattedRecordLength();
  RT_API_ATTRS void DoImpliedEndfile(IoErrorHandler &);
  RT_API_ATTRS void DoEndfile(IoErrorHandler &);
  RT_API_ATTRS void CommitWrites();
  RT_API_ATTRS bool CheckDirectAccess(IoErrorHandler &);
  RT_API_ATTRS void HitEndOnRead(IoErrorHandler &);
  RT_API_ATTRS std::int32_t ReadHeaderOrFooter(std::int64_t frameOffset);

  Lock lock_;

  int unitNumber_{-1};
  Direction direction_{Direction::Output};
  bool impliedEndfile_{false}; // sequential/stream output has taken place
  bool beganReadingRecord_{false};
  bool anyWriteSinceLastPositioning_{false};
  bool directAccessRecWasSet_{false}; // REC= appeared
  // Subtle: The beginning of the frame can't be allowed to advance
  // during a single list-directed READ due to the possibility of a
  // multi-record CHARACTER value with a "r*" repeat count.  So we
  // manage the frame and the current record therein separately.
  std::int64_t frameOffsetInFile_{0};
  std::size_t recordOffsetInFrame_{0}; // of currentRecordNumber
  bool swapEndianness_{false};
  bool createdForInternalChildIo_{false};
  common::BitSet<64> asyncIdAvailable_[maxAsyncIds / 64];

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
  Fortran::common::optional<IoStatementState> io_;

  // A stack of child I/O pseudo-units for defined I/O that have this
  // unit number.
  OwningPtr<ChildIo> child_;
};

// A pseudo-unit for child I/O statements in defined I/O subroutines;
// it forwards operations to the parent I/O statement, which might also
// be a child I/O statement.
class ChildIo {
public:
  RT_API_ATTRS ChildIo(IoStatementState &parent, OwningPtr<ChildIo> &&previous)
      : parent_{parent}, previous_{std::move(previous)} {}

  RT_API_ATTRS IoStatementState &parent() const { return parent_; }

  RT_API_ATTRS void EndIoStatement();

  template <typename A, typename... X>
  RT_API_ATTRS IoStatementState &BeginIoStatement(X &&...xs) {
    RT_DIAG_PUSH
    RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN
    A &state{u_.emplace<A>(std::forward<X>(xs)...)};
    RT_DIAG_POP
    io_.emplace(state);
    return *io_;
  }

  RT_API_ATTRS OwningPtr<ChildIo> AcquirePrevious() {
    return std::move(previous_);
  }

  RT_API_ATTRS Iostat CheckFormattingAndDirection(bool unformatted, Direction);

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
  Fortran::common::optional<IoStatementState> io_;
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_UNIT_H_
