//===-- runtime/unit.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of ExternalFileUnit common for both
// RT_USE_PSEUDO_FILE_UNIT=0 and RT_USE_PSEUDO_FILE_UNIT=1.
//
//===----------------------------------------------------------------------===//
#include "unit.h"
#include "io-error.h"
#include "lock.h"
#include "tools.h"
#include <limits>
#include <utility>

namespace Fortran::runtime::io {

#ifndef FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS
RT_OFFLOAD_VAR_GROUP_BEGIN
RT_VAR_ATTRS ExternalFileUnit *defaultInput{nullptr}; // unit 5
RT_VAR_ATTRS ExternalFileUnit *defaultOutput{nullptr}; // unit 6
RT_VAR_ATTRS ExternalFileUnit *errorOutput{nullptr}; // unit 0 extension
RT_OFFLOAD_VAR_GROUP_END
#endif // FLANG_RUNTIME_NO_GLOBAL_VAR_DEFS

RT_OFFLOAD_API_GROUP_BEGIN

static inline RT_API_ATTRS void SwapEndianness(
    char *data, std::size_t bytes, std::size_t elementBytes) {
  if (elementBytes > 1) {
    auto half{elementBytes >> 1};
    for (std::size_t j{0}; j + elementBytes <= bytes; j += elementBytes) {
      for (std::size_t k{0}; k < half; ++k) {
        RT_DIAG_PUSH
        RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN
        std::swap(data[j + k], data[j + elementBytes - 1 - k]);
        RT_DIAG_POP
      }
    }
  }
}

bool ExternalFileUnit::Emit(const char *data, std::size_t bytes,
    std::size_t elementBytes, IoErrorHandler &handler) {
  auto furthestAfter{std::max(furthestPositionInRecord,
      positionInRecord + static_cast<std::int64_t>(bytes))};
  if (openRecl) {
    // Check for fixed-length record overrun, but allow for
    // sequential record termination.
    int extra{0};
    int header{0};
    if (access == Access::Sequential) {
      if (isUnformatted.value_or(false)) {
        // record header + footer
        header = static_cast<int>(sizeof(std::uint32_t));
        extra = 2 * header;
      } else {
#ifdef _WIN32
        if (!isWindowsTextFile()) {
          ++extra; // carriage return (CR)
        }
#endif
        ++extra; // newline (LF)
      }
    }
    if (furthestAfter > extra + *openRecl) {
      handler.SignalError(IostatRecordWriteOverrun,
          "Attempt to write %zd bytes to position %jd in a fixed-size record "
          "of %jd bytes",
          bytes, static_cast<std::intmax_t>(positionInRecord - header),
          static_cast<std::intmax_t>(*openRecl));
      return false;
    }
  }
  if (recordLength) {
    // It is possible for recordLength to have a value now for a
    // variable-length output record if the previous operation
    // was a BACKSPACE or non advancing input statement.
    recordLength.reset();
    beganReadingRecord_ = false;
  }
  if (IsAfterEndfile()) {
    handler.SignalError(IostatWriteAfterEndfile);
    return false;
  }
  CheckDirectAccess(handler);
  WriteFrame(frameOffsetInFile_, recordOffsetInFrame_ + furthestAfter, handler);
  if (positionInRecord > furthestPositionInRecord) {
    std::memset(Frame() + recordOffsetInFrame_ + furthestPositionInRecord, ' ',
        positionInRecord - furthestPositionInRecord);
  }
  char *to{Frame() + recordOffsetInFrame_ + positionInRecord};
  std::memcpy(to, data, bytes);
  if (swapEndianness_) {
    SwapEndianness(to, bytes, elementBytes);
  }
  positionInRecord += bytes;
  furthestPositionInRecord = furthestAfter;
  anyWriteSinceLastPositioning_ = true;
  return true;
}

bool ExternalFileUnit::Receive(char *data, std::size_t bytes,
    std::size_t elementBytes, IoErrorHandler &handler) {
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
    if (swapEndianness_) {
      SwapEndianness(data, bytes, elementBytes);
    }
    positionInRecord += bytes;
    furthestPositionInRecord = furthestAfter;
    return true;
  } else {
    HitEndOnRead(handler);
    return false;
  }
}

std::size_t ExternalFileUnit::GetNextInputBytes(
    const char *&p, IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, direction_ == Direction::Input);
  std::size_t length{1};
  if (auto recl{EffectiveRecordLength()}) {
    if (positionInRecord < *recl) {
      length = *recl - positionInRecord;
    } else {
      p = nullptr;
      return 0;
    }
  }
  p = FrameNextInput(handler, length);
  return p ? length : 0;
}

std::size_t ExternalFileUnit::ViewBytesInRecord(
    const char *&p, bool forward) const {
  p = nullptr;
  auto recl{recordLength.value_or(positionInRecord)};
  if (forward) {
    if (positionInRecord < recl) {
      p = Frame() + recordOffsetInFrame_ + positionInRecord;
      return recl - positionInRecord;
    }
  } else {
    if (positionInRecord <= recl) {
      p = Frame() + recordOffsetInFrame_ + positionInRecord;
    }
    return positionInRecord - leftTabLimit.value_or(0);
  }
  return 0;
}

const char *ExternalFileUnit::FrameNextInput(
    IoErrorHandler &handler, std::size_t bytes) {
  RUNTIME_CHECK(handler, isUnformatted.has_value() && !*isUnformatted);
  if (static_cast<std::int64_t>(positionInRecord + bytes) <=
      recordLength.value_or(positionInRecord + bytes)) {
    auto at{recordOffsetInFrame_ + positionInRecord};
    auto need{static_cast<std::size_t>(at + bytes)};
    auto got{ReadFrame(frameOffsetInFile_, need, handler)};
    SetVariableFormattedRecordLength();
    if (got >= need) {
      return Frame() + at;
    }
    HitEndOnRead(handler);
  }
  return nullptr;
}

bool ExternalFileUnit::SetVariableFormattedRecordLength() {
  if (recordLength || access == Access::Direct) {
    return true;
  } else if (FrameLength() > recordOffsetInFrame_) {
    const char *record{Frame() + recordOffsetInFrame_};
    std::size_t bytes{FrameLength() - recordOffsetInFrame_};
    if (const char *nl{FindCharacter(record, '\n', bytes)}) {
      recordLength = nl - record;
      if (*recordLength > 0 && record[*recordLength - 1] == '\r') {
        --*recordLength;
      }
      return true;
    }
  }
  return false;
}

bool ExternalFileUnit::BeginReadingRecord(IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, direction_ == Direction::Input);
  if (!beganReadingRecord_) {
    beganReadingRecord_ = true;
    // Don't use IsAtEOF() to check for an EOF condition here, just detect
    // it from a failed or short read from the file.  IsAtEOF() could be
    // wrong for formatted input if actual newline characters had been
    // written in-band by previous WRITEs before a REWIND.  In fact,
    // now that we know that the unit is being used for input (again),
    // it's best to reset endfileRecordNumber and ensure IsAtEOF() will
    // now be true on return only if it gets set by HitEndOnRead().
    endfileRecordNumber.reset();
    if (access == Access::Direct) {
      CheckDirectAccess(handler);
      auto need{static_cast<std::size_t>(recordOffsetInFrame_ + *openRecl)};
      auto got{ReadFrame(frameOffsetInFile_, need, handler)};
      if (got >= need) {
        recordLength = openRecl;
      } else {
        recordLength.reset();
        HitEndOnRead(handler);
      }
    } else {
      if (anyWriteSinceLastPositioning_ && access == Access::Sequential) {
        // Most Fortran implementations allow a READ after a WRITE;
        // the read then just hits an EOF.
        DoEndfile<false, Direction::Input>(handler);
      }
      recordLength.reset();
      RUNTIME_CHECK(handler, isUnformatted.has_value());
      if (*isUnformatted) {
        if (access == Access::Sequential) {
          BeginSequentialVariableUnformattedInputRecord(handler);
        }
      } else { // formatted sequential or stream
        BeginVariableFormattedInputRecord(handler);
      }
    }
  }
  RUNTIME_CHECK(handler,
      recordLength.has_value() || !IsRecordFile() || handler.InError());
  return !handler.InError();
}

void ExternalFileUnit::FinishReadingRecord(IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, direction_ == Direction::Input && beganReadingRecord_);
  beganReadingRecord_ = false;
  if (handler.GetIoStat() == IostatEnd ||
      (IsRecordFile() && !recordLength.has_value())) {
    // Avoid bogus crashes in END/ERR circumstances; but
    // still increment the current record number so that
    // an attempted read of an endfile record, followed by
    // a BACKSPACE, will still be at EOF.
    ++currentRecordNumber;
  } else if (IsRecordFile()) {
    recordOffsetInFrame_ += *recordLength;
    if (access != Access::Direct) {
      RUNTIME_CHECK(handler, isUnformatted.has_value());
      recordLength.reset();
      if (isUnformatted.value_or(false)) {
        // Retain footer in frame for more efficient BACKSPACE
        frameOffsetInFile_ += recordOffsetInFrame_;
        recordOffsetInFrame_ = sizeof(std::uint32_t);
      } else { // formatted
        if (FrameLength() > recordOffsetInFrame_ &&
            Frame()[recordOffsetInFrame_] == '\r') {
          ++recordOffsetInFrame_;
        }
        if (FrameLength() > recordOffsetInFrame_ &&
            Frame()[recordOffsetInFrame_] == '\n') {
          ++recordOffsetInFrame_;
        }
        if (!pinnedFrame || mayPosition()) {
          frameOffsetInFile_ += recordOffsetInFrame_;
          recordOffsetInFrame_ = 0;
        }
      }
    }
    ++currentRecordNumber;
  } else { // unformatted stream
    furthestPositionInRecord =
        std::max(furthestPositionInRecord, positionInRecord);
    frameOffsetInFile_ += recordOffsetInFrame_ + furthestPositionInRecord;
    recordOffsetInFrame_ = 0;
  }
  BeginRecord();
  leftTabLimit.reset();
}

bool ExternalFileUnit::AdvanceRecord(IoErrorHandler &handler) {
  if (direction_ == Direction::Input) {
    FinishReadingRecord(handler);
    return BeginReadingRecord(handler);
  } else { // Direction::Output
    bool ok{true};
    RUNTIME_CHECK(handler, isUnformatted.has_value());
    positionInRecord = furthestPositionInRecord;
    if (access == Access::Direct) {
      if (furthestPositionInRecord <
          openRecl.value_or(furthestPositionInRecord)) {
        // Pad remainder of fixed length record
        WriteFrame(
            frameOffsetInFile_, recordOffsetInFrame_ + *openRecl, handler);
        std::memset(Frame() + recordOffsetInFrame_ + furthestPositionInRecord,
            isUnformatted.value_or(false) ? 0 : ' ',
            *openRecl - furthestPositionInRecord);
        furthestPositionInRecord = *openRecl;
      }
    } else if (*isUnformatted) {
      if (access == Access::Sequential) {
        // Append the length of a sequential unformatted variable-length record
        // as its footer, then overwrite the reserved first four bytes of the
        // record with its length as its header.  These four bytes were skipped
        // over in BeginUnformattedIO<Output>().
        // TODO: Break very large records up into subrecords with negative
        // headers &/or footers
        std::uint32_t length;
        length = furthestPositionInRecord - sizeof length;
        ok = ok &&
            Emit(reinterpret_cast<const char *>(&length), sizeof length,
                sizeof length, handler);
        positionInRecord = 0;
        ok = ok &&
            Emit(reinterpret_cast<const char *>(&length), sizeof length,
                sizeof length, handler);
      } else {
        // Unformatted stream: nothing to do
      }
    } else if (handler.GetIoStat() != IostatOk &&
        furthestPositionInRecord == 0) {
      // Error in formatted variable length record, and no output yet; do
      // nothing, like most other Fortran compilers do.
      return true;
    } else {
      // Terminate formatted variable length record
      const char *lineEnding{"\n"};
      std::size_t lineEndingBytes{1};
#ifdef _WIN32
      if (!isWindowsTextFile()) {
        lineEnding = "\r\n";
        lineEndingBytes = 2;
      }
#endif
      ok = ok && Emit(lineEnding, lineEndingBytes, 1, handler);
    }
    leftTabLimit.reset();
    if (IsAfterEndfile()) {
      return false;
    }
    CommitWrites();
    ++currentRecordNumber;
    if (access != Access::Direct) {
      impliedEndfile_ = IsRecordFile();
      if (IsAtEOF()) {
        endfileRecordNumber.reset();
      }
    }
    return ok;
  }
}

void ExternalFileUnit::BackspaceRecord(IoErrorHandler &handler) {
  if (access == Access::Direct || !IsRecordFile()) {
    handler.SignalError(IostatBackspaceNonSequential,
        "BACKSPACE(UNIT=%d) on direct-access file or unformatted stream",
        unitNumber());
  } else {
    if (IsAfterEndfile()) {
      // BACKSPACE after explicit ENDFILE
      currentRecordNumber = *endfileRecordNumber;
    } else if (leftTabLimit && direction_ == Direction::Input) {
      // BACKSPACE after non-advancing input
      leftTabLimit.reset();
    } else {
      DoImpliedEndfile(handler);
      if (frameOffsetInFile_ + recordOffsetInFrame_ > 0) {
        --currentRecordNumber;
        if (openRecl && access == Access::Direct) {
          BackspaceFixedRecord(handler);
        } else {
          RUNTIME_CHECK(handler, isUnformatted.has_value());
          if (isUnformatted.value_or(false)) {
            BackspaceVariableUnformattedRecord(handler);
          } else {
            BackspaceVariableFormattedRecord(handler);
          }
        }
      }
    }
    BeginRecord();
    anyWriteSinceLastPositioning_ = false;
  }
}

void ExternalFileUnit::FlushOutput(IoErrorHandler &handler) {
  if (!mayPosition()) {
    auto frameAt{FrameAt()};
    if (frameOffsetInFile_ >= frameAt &&
        frameOffsetInFile_ <
            static_cast<std::int64_t>(frameAt + FrameLength())) {
      // A Flush() that's about to happen to a non-positionable file
      // needs to advance frameOffsetInFile_ to prevent attempts at
      // impossible seeks
      CommitWrites();
      leftTabLimit.reset();
    }
  }
  Flush(handler);
}

void ExternalFileUnit::FlushIfTerminal(IoErrorHandler &handler) {
  if (isTerminal()) {
    FlushOutput(handler);
  }
}

void ExternalFileUnit::Endfile(IoErrorHandler &handler) {
  if (access == Access::Direct) {
    handler.SignalError(IostatEndfileDirect,
        "ENDFILE(UNIT=%d) on direct-access file", unitNumber());
  } else if (!mayWrite()) {
    handler.SignalError(IostatEndfileUnwritable,
        "ENDFILE(UNIT=%d) on read-only file", unitNumber());
  } else if (IsAfterEndfile()) {
    // ENDFILE after ENDFILE
  } else {
    DoEndfile(handler);
    if (IsRecordFile() && access != Access::Direct) {
      // Explicit ENDFILE leaves position *after* the endfile record
      RUNTIME_CHECK(handler, endfileRecordNumber.has_value());
      currentRecordNumber = *endfileRecordNumber + 1;
    }
  }
}

void ExternalFileUnit::Rewind(IoErrorHandler &handler) {
  if (access == Access::Direct) {
    handler.SignalError(IostatRewindNonSequential,
        "REWIND(UNIT=%d) on non-sequential file", unitNumber());
  } else {
    DoImpliedEndfile(handler);
    SetPosition(0, handler);
    currentRecordNumber = 1;
    leftTabLimit.reset();
    anyWriteSinceLastPositioning_ = false;
  }
}

void ExternalFileUnit::SetPosition(std::int64_t pos, IoErrorHandler &handler) {
  frameOffsetInFile_ = pos;
  recordOffsetInFrame_ = 0;
  if (access == Access::Direct) {
    directAccessRecWasSet_ = true;
  }
  BeginRecord();
}

bool ExternalFileUnit::SetStreamPos(
    std::int64_t oneBasedPos, IoErrorHandler &handler) {
  if (access != Access::Stream) {
    handler.SignalError("POS= may not appear unless ACCESS='STREAM'");
    return false;
  }
  if (oneBasedPos < 1) { // POS=1 is beginning of file (12.6.2.11)
    handler.SignalError(
        "POS=%zd is invalid", static_cast<std::intmax_t>(oneBasedPos));
    return false;
  }
  // A backwards POS= implies truncation after writing, at least in
  // Intel and NAG.
  if (static_cast<std::size_t>(oneBasedPos - 1) <
      frameOffsetInFile_ + recordOffsetInFrame_) {
    DoImpliedEndfile(handler);
  }
  SetPosition(oneBasedPos - 1, handler);
  // We no longer know which record we're in.  Set currentRecordNumber to
  // a large value from whence we can both advance and backspace.
  currentRecordNumber = std::numeric_limits<std::int64_t>::max() / 2;
  endfileRecordNumber.reset();
  return true;
}

bool ExternalFileUnit::SetDirectRec(
    std::int64_t oneBasedRec, IoErrorHandler &handler) {
  if (access != Access::Direct) {
    handler.SignalError("REC= may not appear unless ACCESS='DIRECT'");
    return false;
  }
  if (!openRecl) {
    handler.SignalError("RECL= was not specified");
    return false;
  }
  if (oneBasedRec < 1) {
    handler.SignalError(
        "REC=%zd is invalid", static_cast<std::intmax_t>(oneBasedRec));
    return false;
  }
  currentRecordNumber = oneBasedRec;
  SetPosition((oneBasedRec - 1) * *openRecl, handler);
  return true;
}

void ExternalFileUnit::EndIoStatement() {
  io_.reset();
  u_.emplace<std::monostate>();
  lock_.Drop();
}

void ExternalFileUnit::BeginSequentialVariableUnformattedInputRecord(
    IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, access == Access::Sequential);
  std::int32_t header{0}, footer{0};
  std::size_t need{recordOffsetInFrame_ + sizeof header};
  std::size_t got{ReadFrame(frameOffsetInFile_, need, handler)};
  // Try to emit informative errors to help debug corrupted files.
  const char *error{nullptr};
  if (got < need) {
    if (got == recordOffsetInFrame_) {
      HitEndOnRead(handler);
    } else {
      error = "Unformatted variable-length sequential file input failed at "
              "record #%jd (file offset %jd): truncated record header";
    }
  } else {
    header = ReadHeaderOrFooter(recordOffsetInFrame_);
    recordLength = sizeof header + header; // does not include footer
    need = recordOffsetInFrame_ + *recordLength + sizeof footer;
    got = ReadFrame(frameOffsetInFile_, need, handler);
    if (got < need) {
      error = "Unformatted variable-length sequential file input failed at "
              "record #%jd (file offset %jd): hit EOF reading record with "
              "length %jd bytes";
    } else {
      footer = ReadHeaderOrFooter(recordOffsetInFrame_ + *recordLength);
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

void ExternalFileUnit::BeginVariableFormattedInputRecord(
    IoErrorHandler &handler) {
  if (this == defaultInput) {
    if (defaultOutput) {
      defaultOutput->FlushOutput(handler);
    }
    if (errorOutput) {
      errorOutput->FlushOutput(handler);
    }
  }
  std::size_t length{0};
  do {
    std::size_t need{length + 1};
    length =
        ReadFrame(frameOffsetInFile_, recordOffsetInFrame_ + need, handler) -
        recordOffsetInFrame_;
    if (length < need) {
      if (length > 0) {
        // final record w/o \n
        recordLength = length;
        unterminatedRecord = true;
      } else {
        HitEndOnRead(handler);
      }
      break;
    }
  } while (!SetVariableFormattedRecordLength());
}

void ExternalFileUnit::BackspaceFixedRecord(IoErrorHandler &handler) {
  RUNTIME_CHECK(handler, openRecl.has_value());
  if (frameOffsetInFile_ < *openRecl) {
    handler.SignalError(IostatBackspaceAtFirstRecord);
  } else {
    frameOffsetInFile_ -= *openRecl;
  }
}

void ExternalFileUnit::BackspaceVariableUnformattedRecord(
    IoErrorHandler &handler) {
  std::int32_t header{0};
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
  if (static_cast<std::int64_t>(got) < headerBytes) {
    handler.SignalError(IostatShortRead);
    return;
  }
  recordLength = ReadHeaderOrFooter(0);
  if (frameOffsetInFile_ < *recordLength + 2 * headerBytes) {
    handler.SignalError(IostatBadUnformattedRecord);
    return;
  }
  frameOffsetInFile_ -= *recordLength + 2 * headerBytes;
  auto need{static_cast<std::size_t>(
      recordOffsetInFrame_ + sizeof header + *recordLength)};
  got = ReadFrame(frameOffsetInFile_, need, handler);
  if (got < need) {
    handler.SignalError(IostatShortRead);
    return;
  }
  header = ReadHeaderOrFooter(recordOffsetInFrame_);
  if (header != *recordLength) {
    handler.SignalError(IostatBadUnformattedRecord);
    return;
  }
}

// There's no portable memrchr(), unfortunately, and strrchr() would
// fail on a record with a NUL, so we have to do it the hard way.
static RT_API_ATTRS const char *FindLastNewline(
    const char *str, std::size_t length) {
  for (const char *p{str + length}; p >= str; p--) {
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
        recordLength = prevNL - (frameOffsetInFile_ + recordOffsetInFrame_);
        break;
      }
    }
    if (frameOffsetInFile_ == 0) {
      recordOffsetInFrame_ = 0;
      recordLength = prevNL;
      break;
    }
    frameOffsetInFile_ -= std::min<std::int64_t>(frameOffsetInFile_, 1024);
    auto need{static_cast<std::size_t>(prevNL + 1 - frameOffsetInFile_)};
    auto got{ReadFrame(frameOffsetInFile_, need, handler)};
    if (got < need) {
      handler.SignalError(IostatShortRead);
      return;
    }
  }
  if (Frame()[recordOffsetInFrame_ + *recordLength] != '\n') {
    handler.SignalError(IostatMissingTerminator);
    return;
  }
  if (*recordLength > 0 &&
      Frame()[recordOffsetInFrame_ + *recordLength - 1] == '\r') {
    --*recordLength;
  }
}

void ExternalFileUnit::DoImpliedEndfile(IoErrorHandler &handler) {
  if (access != Access::Direct) {
    if (!impliedEndfile_ && leftTabLimit && direction_ == Direction::Output) {
      // Flush a partial record after non-advancing output
      impliedEndfile_ = true;
    }
    if (impliedEndfile_ && mayPosition()) {
      DoEndfile(handler);
    }
  }
  impliedEndfile_ = false;
}

template <bool ANY_DIR, Direction DIR>
void ExternalFileUnit::DoEndfile(IoErrorHandler &handler) {
  if (IsRecordFile() && access != Access::Direct) {
    furthestPositionInRecord =
        std::max(positionInRecord, furthestPositionInRecord);
    if (leftTabLimit) { // last I/O was non-advancing
      if (access == Access::Sequential && direction_ == Direction::Output) {
        if constexpr (ANY_DIR || DIR == Direction::Output) {
          // When DoEndfile() is called from BeginReadingRecord(),
          // this call to AdvanceRecord() may appear as a recursion
          // though it may never happen. Expose the call only
          // under the constexpr direction check.
          AdvanceRecord(handler);
        } else {
          // This check always fails if we are here.
          RUNTIME_CHECK(handler, direction_ != Direction::Output);
        }
      } else { // Access::Stream or input
        leftTabLimit.reset();
        ++currentRecordNumber;
      }
    }
    endfileRecordNumber = currentRecordNumber;
  }
  frameOffsetInFile_ += recordOffsetInFrame_ + furthestPositionInRecord;
  recordOffsetInFrame_ = 0;
  FlushOutput(handler);
  Truncate(frameOffsetInFile_, handler);
  TruncateFrame(frameOffsetInFile_, handler);
  BeginRecord();
  impliedEndfile_ = false;
  anyWriteSinceLastPositioning_ = false;
}

template void ExternalFileUnit::DoEndfile(IoErrorHandler &handler);
template void ExternalFileUnit::DoEndfile<false, Direction::Output>(
    IoErrorHandler &handler);
template void ExternalFileUnit::DoEndfile<false, Direction::Input>(
    IoErrorHandler &handler);

void ExternalFileUnit::CommitWrites() {
  frameOffsetInFile_ +=
      recordOffsetInFrame_ + recordLength.value_or(furthestPositionInRecord);
  recordOffsetInFrame_ = 0;
  BeginRecord();
}

bool ExternalFileUnit::CheckDirectAccess(IoErrorHandler &handler) {
  if (access == Access::Direct) {
    RUNTIME_CHECK(handler, openRecl);
    if (!directAccessRecWasSet_) {
      handler.SignalError(
          "No REC= was specified for a data transfer with ACCESS='DIRECT'");
      return false;
    }
  }
  return true;
}

void ExternalFileUnit::HitEndOnRead(IoErrorHandler &handler) {
  handler.SignalEnd();
  if (IsRecordFile() && access != Access::Direct) {
    endfileRecordNumber = currentRecordNumber;
  }
}

ChildIo &ExternalFileUnit::PushChildIo(IoStatementState &parent) {
  OwningPtr<ChildIo> current{std::move(child_)};
  Terminator &terminator{parent.GetIoErrorHandler()};
  OwningPtr<ChildIo> next{New<ChildIo>{terminator}(parent, std::move(current))};
  child_.reset(next.release());
  return *child_;
}

void ExternalFileUnit::PopChildIo(ChildIo &child) {
  if (child_.get() != &child) {
    child.parent().GetIoErrorHandler().Crash(
        "ChildIo being popped is not top of stack");
  }
  child_.reset(child.AcquirePrevious().release()); // deletes top child
}

std::int32_t ExternalFileUnit::ReadHeaderOrFooter(std::int64_t frameOffset) {
  std::int32_t word;
  char *wordPtr{reinterpret_cast<char *>(&word)};
  std::memcpy(wordPtr, Frame() + frameOffset, sizeof word);
  if (swapEndianness_) {
    SwapEndianness(wordPtr, sizeof word, sizeof word);
  }
  return word;
}

void ChildIo::EndIoStatement() {
  io_.reset();
  u_.emplace<std::monostate>();
}

Iostat ChildIo::CheckFormattingAndDirection(
    bool unformatted, Direction direction) {
  bool parentIsInput{!parent_.get_if<IoDirectionState<Direction::Output>>()};
  bool parentIsFormatted{parentIsInput
          ? parent_.get_if<FormattedIoStatementState<Direction::Input>>() !=
              nullptr
          : parent_.get_if<FormattedIoStatementState<Direction::Output>>() !=
              nullptr};
  bool parentIsUnformatted{!parentIsFormatted};
  if (unformatted != parentIsUnformatted) {
    return unformatted ? IostatUnformattedChildOnFormattedParent
                       : IostatFormattedChildOnUnformattedParent;
  } else if (parentIsInput != (direction == Direction::Input)) {
    return parentIsInput ? IostatChildOutputToInputParent
                         : IostatChildInputFromOutputParent;
  } else {
    return IostatOk;
  }
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
