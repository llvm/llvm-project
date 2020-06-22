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

namespace Fortran::runtime::io {

// The per-unit data structures are created on demand so that Fortran I/O
// should work without a Fortran main program.
static Lock unitMapLock;
static UnitMap *unitMap{nullptr};
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
    int unit, const Terminator &terminator, bool *wasExtant) {
  return GetUnitMap().LookUpOrCreate(unit, terminator, wasExtant);
}

ExternalFileUnit *ExternalFileUnit::LookUpForClose(int unit) {
  return GetUnitMap().LookUpForClose(unit);
}

int ExternalFileUnit::NewUnit(const Terminator &terminator) {
  return GetUnitMap().NewUnit(terminator).unitNumber();
}

void ExternalFileUnit::OpenUnit(OpenStatus status, Position position,
    OwningPtr<char> &&newPath, std::size_t newPathLength,
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
    Flush(handler);
    Close(CloseStatus::Keep, handler);
  }
  set_path(std::move(newPath), newPathLength);
  Open(status, position, handler);
}

void ExternalFileUnit::CloseUnit(CloseStatus status, IoErrorHandler &handler) {
  Flush(handler);
  Close(status, handler);
}

void ExternalFileUnit::DestroyClosed() {
  GetUnitMap().DestroyClosed(*this); // destroys *this
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
  unitMap = &New<UnitMap>{}(terminator);
  ExternalFileUnit &out{ExternalFileUnit::LookUpOrCreate(6, terminator)};
  out.Predefine(1);
  out.set_mayRead(false);
  out.set_mayWrite(true);
  out.set_mayPosition(false);
  defaultOutput = &out;
  ExternalFileUnit &in{ExternalFileUnit::LookUpOrCreate(5, terminator)};
  in.Predefine(0);
  in.set_mayRead(true);
  in.set_mayWrite(false);
  in.set_mayPosition(false);
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

bool ExternalFileUnit::Emit(
    const char *data, std::size_t bytes, IoErrorHandler &handler) {
  auto furthestAfter{std::max(furthestPositionInRecord,
      positionInRecord + static_cast<std::int64_t>(bytes))};
  if (furthestAfter > recordLength.value_or(furthestAfter)) {
    handler.SignalError(IostatRecordWriteOverrun);
    return false;
  }
  WriteFrame(frameOffsetInFile_, recordOffsetInFrame_ + furthestAfter, handler);
  if (positionInRecord > furthestPositionInRecord) {
    std::memset(Frame() + furthestPositionInRecord, ' ',
        positionInRecord - furthestPositionInRecord);
  }
  std::memcpy(Frame() + positionInRecord, data, bytes);
  positionInRecord += bytes;
  furthestPositionInRecord = furthestAfter;
  return true;
}

std::optional<char32_t> ExternalFileUnit::GetCurrentChar(
    IoErrorHandler &handler) {
  isReading_ = true; // TODO: manage read/write transitions
  if (isUnformatted) {
    handler.Crash("GetCurrentChar() called for unformatted input");
    return std::nullopt;
  }
  std::size_t chunk{256}; // for stream input
  if (recordLength.has_value()) {
    if (positionInRecord >= *recordLength) {
      return std::nullopt;
    }
    chunk = *recordLength - positionInRecord;
  }
  auto at{recordOffsetInFrame_ + positionInRecord};
  std::size_t need{static_cast<std::size_t>(at + 1)};
  std::size_t want{need + chunk};
  auto got{ReadFrame(frameOffsetInFile_, want, handler)};
  if (got <= need) {
    endfileRecordNumber = currentRecordNumber;
    handler.SignalEnd();
    return std::nullopt;
  }
  const char *p{Frame() + at};
  if (isUTF8) {
    // TODO: UTF-8 decoding
  }
  return *p;
}

void ExternalFileUnit::SetLeftTabLimit() {
  leftTabLimit = furthestPositionInRecord;
  positionInRecord = furthestPositionInRecord;
}

bool ExternalFileUnit::AdvanceRecord(IoErrorHandler &handler) {
  bool ok{true};
  if (isReading_) {
    if (access == Access::Sequential) {
      if (isUnformatted) {
        NextSequentialUnformattedInputRecord(handler);
      } else {
        NextSequentialFormattedInputRecord(handler);
      }
    }
  } else if (!isUnformatted) {
    if (recordLength.has_value()) {
      // fill fixed-size record
      if (furthestPositionInRecord < *recordLength) {
        WriteFrame(frameOffsetInFile_, *recordLength, handler);
        std::memset(Frame() + recordOffsetInFrame_ + furthestPositionInRecord,
            ' ', *recordLength - furthestPositionInRecord);
      }
    } else {
      positionInRecord = furthestPositionInRecord;
      ok &= Emit("\n", 1, handler); // TODO: Windows CR+LF
      frameOffsetInFile_ += recordOffsetInFrame_ + furthestPositionInRecord;
      recordOffsetInFrame_ = 0;
    }
  }
  ++currentRecordNumber;
  positionInRecord = 0;
  furthestPositionInRecord = 0;
  leftTabLimit.reset();
  return ok;
}

void ExternalFileUnit::BackspaceRecord(IoErrorHandler &handler) {
  if (!isReading_) {
    handler.Crash("ExternalFileUnit::BackspaceRecord() called during writing");
    // TODO: create endfile record, &c.
  }
  if (access == Access::Sequential) {
    if (isUnformatted) {
      BackspaceSequentialUnformattedRecord(handler);
    } else {
      BackspaceSequentialFormattedRecord(handler);
    }
  } else {
    // TODO
  }
  positionInRecord = 0;
  furthestPositionInRecord = 0;
  leftTabLimit.reset();
}

void ExternalFileUnit::FlushIfTerminal(IoErrorHandler &handler) {
  if (isTerminal()) {
    Flush(handler);
  }
}

void ExternalFileUnit::EndIoStatement() {
  frameOffsetInFile_ += recordOffsetInFrame_;
  recordOffsetInFrame_ = 0;
  io_.reset();
  u_.emplace<std::monostate>();
  lock_.Drop();
}

void ExternalFileUnit::NextSequentialUnformattedInputRecord(
    IoErrorHandler &handler) {
  std::int32_t header{0}, footer{0};
  // Retain previous footer (if any) in frame for more efficient BACKSPACE
  std::size_t retain{sizeof header};
  if (recordLength) { // not first record - advance to next
    ++currentRecordNumber;
    if (endfileRecordNumber && currentRecordNumber >= *endfileRecordNumber) {
      handler.SignalEnd();
      return;
    }
    frameOffsetInFile_ +=
        recordOffsetInFrame_ + *recordLength + 2 * sizeof header;
    recordOffsetInFrame_ = 0;
  } else {
    retain = 0;
  }
  std::size_t need{retain + sizeof header};
  std::size_t got{ReadFrame(frameOffsetInFile_ - retain, need, handler)};
  // Try to emit informative errors to help debug corrupted files.
  const char *error{nullptr};
  if (got < need) {
    if (got == retain) {
      handler.SignalEnd();
    } else {
      error = "Unformatted sequential file input failed at record #%jd (file "
              "offset %jd): truncated record header";
    }
  } else {
    std::memcpy(&header, Frame() + retain, sizeof header);
    need = retain + header + 2 * sizeof header;
    got = ReadFrame(frameOffsetInFile_ - retain,
        need + sizeof header /* next one */, handler);
    if (got < need) {
      error = "Unformatted sequential file input failed at record #%jd (file "
              "offset %jd): hit EOF reading record with length %jd bytes";
    } else {
      const char *start{Frame() + retain + sizeof header};
      std::memcpy(&footer, start + header, sizeof footer);
      if (footer != header) {
        error = "Unformatted sequential file input failed at record #%jd (file "
                "offset %jd): record header has length %jd that does not match "
                "record footer (%jd)";
      } else {
        recordLength = header;
      }
    }
  }
  if (error) {
    handler.SignalError(error, static_cast<std::intmax_t>(currentRecordNumber),
        static_cast<std::intmax_t>(frameOffsetInFile_),
        static_cast<std::intmax_t>(header), static_cast<std::intmax_t>(footer));
  }
  positionInRecord = sizeof header;
}

void ExternalFileUnit::NextSequentialFormattedInputRecord(
    IoErrorHandler &handler) {
  static constexpr std::size_t chunk{256};
  std::size_t length{0};
  if (recordLength.has_value()) {
    // not first record - advance to next
    ++currentRecordNumber;
    if (endfileRecordNumber && currentRecordNumber >= *endfileRecordNumber) {
      handler.SignalEnd();
      return;
    }
    if (Frame()[*recordLength] == '\r') {
      ++*recordLength;
    }
    recordOffsetInFrame_ += *recordLength + 1;
  }
  while (true) {
    std::size_t got{ReadFrame(
        frameOffsetInFile_, recordOffsetInFrame_ + length + chunk, handler)};
    if (got <= recordOffsetInFrame_ + length) {
      handler.SignalEnd();
      break;
    }
    const char *frame{Frame() + recordOffsetInFrame_};
    if (const char *nl{reinterpret_cast<const char *>(
            std::memchr(frame + length, '\n', chunk))}) {
      recordLength = nl - (frame + length) + 1;
      if (*recordLength > 0 && frame[*recordLength - 1] == '\r') {
        --*recordLength;
      }
      return;
    }
    length += got;
  }
}

void ExternalFileUnit::BackspaceSequentialUnformattedRecord(
    IoErrorHandler &handler) {
  std::int32_t header{0}, footer{0};
  RUNTIME_CHECK(handler, currentRecordNumber > 1);
  --currentRecordNumber;
  int overhead{static_cast<int>(2 * sizeof header)};
  // Error conditions here cause crashes, not file format errors, because the
  // validity of the file structure before the current record will have been
  // checked informatively in NextSequentialUnformattedInputRecord().
  RUNTIME_CHECK(handler, frameOffsetInFile_ >= overhead);
  std::size_t got{
      ReadFrame(frameOffsetInFile_ - sizeof footer, sizeof footer, handler)};
  RUNTIME_CHECK(handler, got >= sizeof footer);
  std::memcpy(&footer, Frame(), sizeof footer);
  RUNTIME_CHECK(handler, frameOffsetInFile_ >= footer + overhead);
  frameOffsetInFile_ -= footer + 2 * sizeof footer;
  auto extra{std::max<std::size_t>(sizeof footer, frameOffsetInFile_)};
  std::size_t want{extra + footer + 2 * sizeof footer};
  got = ReadFrame(frameOffsetInFile_ - extra, want, handler);
  RUNTIME_CHECK(handler, got >= want);
  std::memcpy(&header, Frame() + extra, sizeof header);
  RUNTIME_CHECK(handler, header == footer);
  positionInRecord = sizeof header;
  recordLength = footer;
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

void ExternalFileUnit::BackspaceSequentialFormattedRecord(
    IoErrorHandler &handler) {
  std::int64_t start{frameOffsetInFile_ + recordOffsetInFrame_};
  --currentRecordNumber;
  RUNTIME_CHECK(handler, currentRecordNumber > 0);
  if (currentRecordNumber == 1) {
    // To simplify the code below, treat a backspace to the first record
    // as a special case;
    RUNTIME_CHECK(handler, start > 0);
    *recordLength = start - 1;
    frameOffsetInFile_ = 0;
    recordOffsetInFrame_ = 0;
    ReadFrame(0, *recordLength + 1, handler);
  } else {
    RUNTIME_CHECK(handler, start > 1);
    std::int64_t at{start - 2}; // byte before previous record's newline
    while (true) {
      if (const char *p{
              FindLastNewline(Frame(), at - frameOffsetInFile_ + 1)}) {
        // This is the newline that ends the record before the previous one.
        recordOffsetInFrame_ = p - Frame() + 1;
        *recordLength = start - 1 - (frameOffsetInFile_ + recordOffsetInFrame_);
        break;
      }
      RUNTIME_CHECK(handler, frameOffsetInFile_ > 0);
      at = frameOffsetInFile_ - 1;
      if (auto bytesBefore{BytesBufferedBeforeFrame()}) {
        frameOffsetInFile_ = FrameAt() - bytesBefore;
      } else {
        static constexpr int chunk{1024};
        frameOffsetInFile_ = std::max<std::int64_t>(0, at - chunk);
      }
      std::size_t want{static_cast<std::size_t>(start - frameOffsetInFile_)};
      std::size_t got{ReadFrame(frameOffsetInFile_, want, handler)};
      RUNTIME_CHECK(handler, got >= want);
    }
  }
  std::size_t want{
      static_cast<std::size_t>(recordOffsetInFrame_ + *recordLength + 1)};
  RUNTIME_CHECK(handler, FrameLength() >= want);
  RUNTIME_CHECK(handler, Frame()[recordOffsetInFrame_ + *recordLength] == '\n');
  if (*recordLength > 0 &&
      Frame()[recordOffsetInFrame_ + *recordLength - 1] == '\r') {
    --*recordLength;
  }
}
} // namespace Fortran::runtime::io
