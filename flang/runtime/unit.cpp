//===-- runtime/unit.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "unit.h"
#include "io-error.h"
#include "lock.h"
#include "unit-map.h"
#include <cstdio>

namespace Fortran::runtime::io {

// The per-unit data structures are created on demand so that Fortran I/O
// should work without a Fortran main program.
static Lock unitMapLock;
static UnitMap *unitMap{nullptr};
static ExternalFileUnit *defaultInput{nullptr};
static ExternalFileUnit *defaultOutput{nullptr};

void FlushOutputOnCrash(const Terminator &terminator) {
  if (!defaultOutput) {
    return;
  }
  CriticalSection critical{unitMapLock};
  if (defaultOutput) {
    IoErrorHandler handler{terminator};
    handler.HasIoStat(); // prevent nested crash if flush has error
    defaultOutput->Flush(handler);
  }
}

ExternalFileUnit *ExternalFileUnit::LookUp(int unit) {
  return GetUnitMap().LookUp(unit);
}

ExternalFileUnit &ExternalFileUnit::LookUpOrCrash(
    int unit, const Terminator &terminator) {
  ExternalFileUnit *file{LookUp(unit)};
  if (!file) {
    terminator.Crash("Not an open I/O unit number: %d", unit);
  }
  return *file;
}

ExternalFileUnit &ExternalFileUnit::LookUpOrCreate(
    int unit, const Terminator &terminator, bool &wasExtant) {
  return GetUnitMap().LookUpOrCreate(unit, terminator, wasExtant);
}

ExternalFileUnit &ExternalFileUnit::LookUpOrCreateAnonymous(
    int unit, Direction dir, bool isUnformatted, const Terminator &terminator) {
  bool exists{false};
  ExternalFileUnit &result{
      GetUnitMap().LookUpOrCreate(unit, terminator, exists)};
  if (!exists) {
    // I/O to an unconnected unit reads/creates a local file, e.g. fort.7
    std::size_t pathMaxLen{32};
    auto path{SizedNew<char>{terminator}(pathMaxLen)};
    std::snprintf(path.get(), pathMaxLen, "fort.%d", unit);
    IoErrorHandler handler{terminator};
    result.OpenUnit(
        dir == Direction::Input ? OpenStatus::Old : OpenStatus::Replace,
        Action::ReadWrite, Position::Rewind, std::move(path),
        std::strlen(path.get()), handler);
    result.isUnformatted = isUnformatted;
  }
  return result;
}

ExternalFileUnit &ExternalFileUnit::CreateNew(
    int unit, const Terminator &terminator) {
  bool wasExtant{false};
  ExternalFileUnit &result{
      GetUnitMap().LookUpOrCreate(unit, terminator, wasExtant)};
  RUNTIME_CHECK(terminator, !wasExtant);
  return result;
}

ExternalFileUnit *ExternalFileUnit::LookUpForClose(int unit) {
  return GetUnitMap().LookUpForClose(unit);
}

int ExternalFileUnit::NewUnit(const Terminator &terminator) {
  return GetUnitMap().NewUnit(terminator).unitNumber();
}

void ExternalFileUnit::OpenUnit(OpenStatus status, std::optional<Action> action,
    Position position, OwningPtr<char> &&newPath, std::size_t newPathLength,
    IoErrorHandler &handler) {
  if (IsOpen()) {
    if (status == OpenStatus::Old &&
        (!newPath.get() ||
            (path() && pathLength() == newPathLength &&
                std::memcmp(path(), newPath.get(), newPathLength) == 0))) {
      // OPEN of existing unit, STATUS='OLD', not new FILE=
      newPath.reset();
      return;
    }
    // Otherwise, OPEN on open unit with new FILE= implies CLOSE
    DoImpliedEndfile(handler);
    Flush(handler);
    Close(CloseStatus::Keep, handler);
  }
  set_path(std::move(newPath), newPathLength);
  Open(status, action, position, handler);
  auto totalBytes{knownSize()};
  if (access == Access::Direct) {
    if (!isFixedRecordLength || !recordLength) {
      handler.SignalError(IostatOpenBadRecl,
          "OPEN(UNIT=%d,ACCESS='DIRECT'): record length is not known",
          unitNumber());
    } else if (*recordLength <= 0) {
      handler.SignalError(IostatOpenBadRecl,
          "OPEN(UNIT=%d,ACCESS='DIRECT',RECL=%jd): record length is invalid",
          unitNumber(), static_cast<std::intmax_t>(*recordLength));
    } else if (!totalBytes) {
      handler.SignalError(IostatOpenUnknownSize,
          "OPEN(UNIT=%d,ACCESS='DIRECT'): file size is not known");
    } else if (*totalBytes % *recordLength != 0) {
      handler.SignalError(IostatOpenBadAppend,
          "OPEN(UNIT=%d,ACCESS='DIRECT',RECL=%jd): record length is not an "
          "even divisor of the file size %jd",
          unitNumber(), static_cast<std::intmax_t>(*recordLength),
          static_cast<std::intmax_t>(*totalBytes));
    }
  }
  if (position == Position::Append) {
    if (*totalBytes && recordLength && *recordLength) {
      endfileRecordNumber = 1 + (*totalBytes / *recordLength);
    } else {
      // Fake it so that we can backspace relative from the end
      endfileRecordNumber = std::numeric_limits<std::int64_t>::max() - 1;
    }
    currentRecordNumber = *endfileRecordNumber;
  } else {
    currentRecordNumber = 1;
  }
}

void ExternalFileUnit::CloseUnit(CloseStatus status, IoErrorHandler &handler) {
  DoImpliedEndfile(handler);
  Flush(handler);
  Close(status, handler);
}

void ExternalFileUnit::DestroyClosed() {
  GetUnitMap().DestroyClosed(*this); // destroys *this
}

bool ExternalFileUnit::SetDirection(
    Direction direction, IoErrorHandler &handler) {
  if (direction == Direction::Input) {
    if (mayRead()) {
      direction_ = Direction::Input;
      return true;
    } else {
      handler.SignalError(IostatReadFromWriteOnly,
          "READ(UNIT=%d) with ACTION='WRITE'", unitNumber());
      return false;
    }
  } else {
    if (mayWrite()) {
      direction_ = Direction::Output;
      return true;
    } else {
      handler.SignalError(IostatWriteToReadOnly,
          "WRITE(UNIT=%d) with ACTION='READ'", unitNumber());
      return false;
    }
  }
}

UnitMap &ExternalFileUnit::GetUnitMap() {
  if (unitMap) {
    return *unitMap;
  }
  CriticalSection critical{unitMapLock};
  if (unitMap) {
    return *unitMap;
  }
  Terminator terminator{__FILE__, __LINE__};
  IoErrorHandler handler{terminator};
  unitMap = New<UnitMap>{terminator}().release();
  ExternalFileUnit &out{ExternalFileUnit::CreateNew(6, terminator)};
  out.Predefine(1);
  out.SetDirection(Direction::Output, handler);
  defaultOutput = &out;
  ExternalFileUnit &in{ExternalFileUnit::CreateNew(5, terminator)};
  in.Predefine(0);
  in.SetDirection(Direction::Input, handler);
  defaultInput = &in;
  // TODO: Set UTF-8 mode from the environment
  return *unitMap;
}

void ExternalFileUnit::CloseAll(IoErrorHandler &handler) {
  CriticalSection critical{unitMapLock};
  if (unitMap) {
    unitMap->CloseAll(handler);
    FreeMemoryAndNullify(unitMap);
  }
  defaultOutput = nullptr;
}

void ExternalFileUnit::FlushAll(IoErrorHandler &handler) {
  CriticalSection critical{unitMapLock};
  if (unitMap) {
    unitMap->FlushAll(handler);
  }
}

bool ExternalFileUnit::Emit(
    const char *data, std::size_t bytes, IoErrorHandler &handler) {
  auto furthestAfter{std::max(furthestPositionInRecord,
      positionInRecord + static_cast<std::int64_t>(bytes))};
  if (furthestAfter > recordLength.value_or(furthestAfter)) {
    handler.SignalError(IostatRecordWriteOverrun,
        "Attempt to write %zd bytes to position %jd in a fixed-size record of "
        "%jd bytes",
        bytes, static_cast<std::intmax_t>(positionInRecord),
        static_cast<std::intmax_t>(*recordLength));
    return false;
  }
  WriteFrame(frameOffsetInFile_, recordOffsetInFrame_ + furthestAfter, handler);
  if (positionInRecord > furthestPositionInRecord) {
    std::memset(Frame() + recordOffsetInFrame_ + furthestPositionInRecord, ' ',
        positionInRecord - furthestPositionInRecord);
  }
  std::memcpy(Frame() + recordOffsetInFrame_ + positionInRecord, data, bytes);
  positionInRecord += bytes;
  furthestPositionInRecord = furthestAfter;
  return true;
}

bool ExternalFileUnit::Receive(
    char *data, std::size_t bytes, IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, direction_ == Direction::Input);
  auto furthestAfter{std::max(furthestPositionInRecord,
      positionInRecord + static_cast<std::int64_t>(bytes))};
  if (furthestAfter > recordLength.value_or(furthestAfter)) {
    handler.SignalError(IostatRecordReadOverrun,
        "Attempt to read %zd bytes at position %jd in a record of %jd bytes",
        bytes, static_cast<std::intmax_t>(positionInRecord),
        static_cast<std::intmax_t>(*recordLength));
    return false;
  }
  auto need{recordOffsetInFrame_ + furthestAfter};
  auto got{ReadFrame(frameOffsetInFile_, need, handler)};
  if (got >= need) {
    std::memcpy(data, Frame() + recordOffsetInFrame_ + positionInRecord, bytes);
    positionInRecord += bytes;
    furthestPositionInRecord = furthestAfter;
    return true;
  } else {
    handler.SignalEnd();
    endfileRecordNumber = currentRecordNumber;
    return false;
  }
}

std::optional<char32_t> ExternalFileUnit::GetCurrentChar(
    IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, direction_ == Direction::Input);
  if (const char *p{FrameNextInput(handler, 1)}) {
    // TODO: UTF-8 decoding; may have to get more bytes in a loop
    return *p;
  }
  return std::nullopt;
}

const char *ExternalFileUnit::FrameNextInput(
    IoErrorHandler &handler, std::size_t bytes) {
  RUNTIME_CHECK(handler, !isUnformatted);
  if (static_cast<std::int64_t>(positionInRecord + bytes) <=
      recordLength.value_or(positionInRecord + bytes)) {
    auto at{recordOffsetInFrame_ + positionInRecord};
    auto need{static_cast<std::size_t>(at + bytes)};
    auto got{ReadFrame(frameOffsetInFile_, need, handler)};
    SetSequentialVariableFormattedRecordLength();
    if (got >= need) {
      return Frame() + at;
    }
    handler.SignalEnd();
    endfileRecordNumber = currentRecordNumber;
  }
  return nullptr;
}

bool ExternalFileUnit::SetSequentialVariableFormattedRecordLength() {
  if (recordLength || access != Access::Sequential) {
    return true;
  }
  if (FrameLength() > recordOffsetInFrame_) {
    const char *record{Frame() + recordOffsetInFrame_};
    if (const char *nl{reinterpret_cast<const char *>(
            std::memchr(record, '\n', FrameLength() - recordOffsetInFrame_))}) {
      recordLength = nl - record;
      if (*recordLength > 0 && record[*recordLength - 1] == '\r') {
        --*recordLength;
      }
      return true;
    }
  }
  return false;
}

void ExternalFileUnit::SetLeftTabLimit() {
  leftTabLimit = furthestPositionInRecord;
  positionInRecord = furthestPositionInRecord;
}

void ExternalFileUnit::BeginReadingRecord(IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, direction_ == Direction::Input);
  if (access == Access::Sequential) {
    if (endfileRecordNumber && currentRecordNumber >= *endfileRecordNumber) {
      handler.SignalEnd();
    } else if (isFixedRecordLength) {
      RUNTIME_CHECK(handler, recordLength.has_value());
      auto need{static_cast<std::size_t>(recordOffsetInFrame_ + *recordLength)};
      auto got{ReadFrame(frameOffsetInFile_, need, handler)};
      if (got < need) {
        handler.SignalEnd();
      }
    } else if (isUnformatted) {
      BeginSequentialVariableUnformattedInputRecord(handler);
    } else { // formatted
      BeginSequentialVariableFormattedInputRecord(handler);
    }
  }
}

bool ExternalFileUnit::AdvanceRecord(IoErrorHandler &handler) {
  bool ok{true};
  if (direction_ == Direction::Input) {
    if (access == Access::Sequential) {
      RUNTIME_CHECK(handler, recordLength.has_value());
      if (isFixedRecordLength) {
        frameOffsetInFile_ += recordOffsetInFrame_ + *recordLength;
        recordOffsetInFrame_ = 0;
      } else if (isUnformatted) {
        // Retain footer in frame for more efficient BACKSPACE
        frameOffsetInFile_ += recordOffsetInFrame_ + *recordLength;
        recordOffsetInFrame_ = sizeof(std::uint32_t);
        recordLength.reset();
      } else { // formatted
        if (Frame()[recordOffsetInFrame_ + *recordLength] == '\r') {
          ++recordOffsetInFrame_;
        }
        recordOffsetInFrame_ += *recordLength + 1;
        RUNTIME_CHECK(handler, Frame()[recordOffsetInFrame_ - 1] == '\n');
        recordLength.reset();
      }
    }
  } else { // Direction::Output
    if (!isUnformatted) {
      if (isFixedRecordLength && recordLength) {
        if (furthestPositionInRecord < *recordLength) {
          WriteFrame(frameOffsetInFile_, *recordLength, handler);
          std::memset(Frame() + recordOffsetInFrame_ + furthestPositionInRecord,
              ' ', *recordLength - furthestPositionInRecord);
        }
      } else {
        positionInRecord = furthestPositionInRecord;
        ok &= Emit("\n", 1, handler); // TODO: Windows CR+LF
      }
    }
    frameOffsetInFile_ +=
        recordOffsetInFrame_ + recordLength.value_or(furthestPositionInRecord);
    recordOffsetInFrame_ = 0;
    impliedEndfile_ = true;
  }
  ++currentRecordNumber;
  BeginRecord();
  return ok;
}

void ExternalFileUnit::BackspaceRecord(IoErrorHandler &handler) {
  if (access != Access::Sequential) {
    handler.SignalError(IostatBackspaceNonSequential,
        "BACKSPACE(UNIT=%d) on non-sequential file", unitNumber());
  } else {
    DoImpliedEndfile(handler);
    --currentRecordNumber;
    BeginRecord();
    if (isFixedRecordLength) {
      BackspaceFixedRecord(handler);
    } else if (isUnformatted) {
      BackspaceVariableUnformattedRecord(handler);
    } else {
      BackspaceVariableFormattedRecord(handler);
    }
  }
}

void ExternalFileUnit::FlushIfTerminal(IoErrorHandler &handler) {
  if (isTerminal()) {
    Flush(handler);
  }
}

void ExternalFileUnit::Endfile(IoErrorHandler &handler) {
  if (access != Access::Sequential) {
    handler.SignalError(IostatEndfileNonSequential,
        "ENDFILE(UNIT=%d) on non-sequential file", unitNumber());
  } else if (!mayWrite()) {
    handler.SignalError(IostatEndfileUnwritable,
        "ENDFILE(UNIT=%d) on read-only file", unitNumber());
  } else {
    DoEndfile(handler);
  }
}

void ExternalFileUnit::Rewind(IoErrorHandler &handler) {
  if (access == Access::Direct) {
    handler.SignalError(IostatRewindNonSequential,
        "REWIND(UNIT=%d) on non-sequential file", unitNumber());
  } else {
    DoImpliedEndfile(handler);
    SetPosition(0);
    currentRecordNumber = 1;
    // TODO: reset endfileRecordNumber?
  }
}

void ExternalFileUnit::EndIoStatement() {
  frameOffsetInFile_ += recordOffsetInFrame_;
  recordOffsetInFrame_ = 0;
  io_.reset();
  u_.emplace<std::monostate>();
  lock_.Drop();
}

void ExternalFileUnit::BeginSequentialVariableUnformattedInputRecord(
    IoErrorHandler &handler) {
  std::int32_t header{0}, footer{0};
  std::size_t need{recordOffsetInFrame_ + sizeof header};
  std::size_t got{ReadFrame(frameOffsetInFile_, need, handler)};
  // Try to emit informative errors to help debug corrupted files.
  const char *error{nullptr};
  if (got < need) {
    if (got == recordOffsetInFrame_) {
      handler.SignalEnd();
    } else {
      error = "Unformatted variable-length sequential file input failed at "
              "record #%jd (file offset %jd): truncated record header";
    }
  } else {
    std::memcpy(&header, Frame() + recordOffsetInFrame_, sizeof header);
    recordLength = sizeof header + header; // does not include footer
    need = recordOffsetInFrame_ + *recordLength + sizeof footer;
    got = ReadFrame(frameOffsetInFile_, need, handler);
    if (got < need) {
      error = "Unformatted variable-length sequential file input failed at "
              "record #%jd (file offset %jd): hit EOF reading record with "
              "length %jd bytes";
    } else {
      std::memcpy(&footer, Frame() + recordOffsetInFrame_ + *recordLength,
          sizeof footer);
      if (footer != header) {
        error = "Unformatted variable-length sequential file input failed at "
                "record #%jd (file offset %jd): record header has length %jd "
                "that does not match record footer (%jd)";
      }
    }
  }
  if (error) {
    handler.SignalError(error, static_cast<std::intmax_t>(currentRecordNumber),
        static_cast<std::intmax_t>(frameOffsetInFile_),
        static_cast<std::intmax_t>(header), static_cast<std::intmax_t>(footer));
    // TODO: error recovery
  }
  positionInRecord = sizeof header;
}

void ExternalFileUnit::BeginSequentialVariableFormattedInputRecord(
    IoErrorHandler &handler) {
  if (this == defaultInput && defaultOutput) {
    defaultOutput->Flush(handler);
  }
  std::size_t length{0};
  do {
    std::size_t need{recordOffsetInFrame_ + length + 1};
    length = ReadFrame(frameOffsetInFile_, need, handler);
    if (length < need) {
      handler.SignalEnd();
      break;
    }
  } while (!SetSequentialVariableFormattedRecordLength());
}

void ExternalFileUnit::BackspaceFixedRecord(IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, recordLength.has_value());
  if (frameOffsetInFile_ < *recordLength) {
    handler.SignalError(IostatBackspaceAtFirstRecord);
  } else {
    frameOffsetInFile_ -= *recordLength;
  }
}

void ExternalFileUnit::BackspaceVariableUnformattedRecord(
    IoErrorHandler &handler) {
  std::int32_t header{0}, footer{0};
  auto headerBytes{static_cast<std::int64_t>(sizeof header)};
  frameOffsetInFile_ += recordOffsetInFrame_;
  recordOffsetInFrame_ = 0;
  if (frameOffsetInFile_ <= headerBytes) {
    handler.SignalError(IostatBackspaceAtFirstRecord);
    return;
  }
  // Error conditions here cause crashes, not file format errors, because the
  // validity of the file structure before the current record will have been
  // checked informatively in NextSequentialVariableUnformattedInputRecord().
  std::size_t got{
      ReadFrame(frameOffsetInFile_ - headerBytes, headerBytes, handler)};
  RUNTIME_CHECK(handler, got >= sizeof footer);
  std::memcpy(&footer, Frame(), sizeof footer);
  recordLength = footer;
  RUNTIME_CHECK(handler, frameOffsetInFile_ >= *recordLength + 2 * headerBytes);
  frameOffsetInFile_ -= *recordLength + 2 * headerBytes;
  if (frameOffsetInFile_ >= headerBytes) {
    frameOffsetInFile_ -= headerBytes;
    recordOffsetInFrame_ = headerBytes;
  }
  auto need{static_cast<std::size_t>(
      recordOffsetInFrame_ + sizeof header + *recordLength)};
  got = ReadFrame(frameOffsetInFile_, need, handler);
  RUNTIME_CHECK(handler, got >= need);
  std::memcpy(&header, Frame() + recordOffsetInFrame_, sizeof header);
  RUNTIME_CHECK(handler, header == *recordLength);
}

// There's no portable memrchr(), unfortunately, and strrchr() would
// fail on a record with a NUL, so we have to do it the hard way.
static const char *FindLastNewline(const char *str, std::size_t length) {
  for (const char *p{str + length}; p-- > str;) {
    if (*p == '\n') {
      return p;
    }
  }
  return nullptr;
}

void ExternalFileUnit::BackspaceVariableFormattedRecord(
    IoErrorHandler &handler) {
  // File offset of previous record's newline
  auto prevNL{
      frameOffsetInFile_ + static_cast<std::int64_t>(recordOffsetInFrame_) - 1};
  if (prevNL < 0) {
    handler.SignalError(IostatBackspaceAtFirstRecord);
    return;
  }
  while (true) {
    if (frameOffsetInFile_ < prevNL) {
      if (const char *p{
              FindLastNewline(Frame(), prevNL - 1 - frameOffsetInFile_)}) {
        recordOffsetInFrame_ = p - Frame() + 1;
        *recordLength = prevNL - (frameOffsetInFile_ + recordOffsetInFrame_);
        break;
      }
    }
    if (frameOffsetInFile_ == 0) {
      recordOffsetInFrame_ = 0;
      *recordLength = prevNL;
      break;
    }
    frameOffsetInFile_ -= std::min<std::int64_t>(frameOffsetInFile_, 1024);
    auto need{static_cast<std::size_t>(prevNL + 1 - frameOffsetInFile_)};
    auto got{ReadFrame(frameOffsetInFile_, need, handler)};
    RUNTIME_CHECK(handler, got >= need);
  }
  RUNTIME_CHECK(handler, Frame()[recordOffsetInFrame_ + *recordLength] == '\n');
  if (*recordLength > 0 &&
      Frame()[recordOffsetInFrame_ + *recordLength - 1] == '\r') {
    --*recordLength;
  }
}

void ExternalFileUnit::DoImpliedEndfile(IoErrorHandler &handler) {
  if (impliedEndfile_) {
    impliedEndfile_ = false;
    if (access == Access::Sequential && mayPosition()) {
      DoEndfile(handler);
    }
  }
}

void ExternalFileUnit::DoEndfile(IoErrorHandler &handler) {
  endfileRecordNumber = currentRecordNumber;
  Truncate(frameOffsetInFile_ + recordOffsetInFrame_, handler);
  BeginRecord();
  impliedEndfile_ = false;
}
} // namespace Fortran::runtime::io
