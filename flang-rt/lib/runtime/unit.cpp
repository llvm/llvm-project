//===-- lib/runtime/unit.cpp ------------------------------------*- C++ -*-===//
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
#include "flang-rt/runtime/unit.h"
#include "flang-rt/runtime/io-error.h"
#include "flang-rt/runtime/lock.h"
#include "flang-rt/runtime/tools.h"
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
#ifndef RT_DEVICE_COMPILATION
  switch (elementBytes) {
  case 2:
    for (std::size_t j{0}; j + 2 <= bytes; j += 2) {
      std::uint16_t x;
      runtime::memcpy(&x, data + j, sizeof(x));
#if defined(_MSC_VER)
      x = _byteswap_ushort(x);
#else
      x = __builtin_bswap16(x);
#endif
      runtime::memcpy(data + j, &x, sizeof(x));
    }
    return;
  case 4:
    for (std::size_t j{0}; j + 4 <= bytes; j += 4) {
      std::uint32_t x;
      runtime::memcpy(&x, data + j, sizeof(x));
#if defined(_MSC_VER)
      x = _byteswap_ulong(x);
#else
      x = __builtin_bswap32(x);
#endif
      runtime::memcpy(data + j, &x, sizeof(x));
    }
    return;
  case 8:
    for (std::size_t j{0}; j + 8 <= bytes; j += 8) {
      std::uint64_t x;
      runtime::memcpy(&x, data + j, sizeof(x));
#if defined(_MSC_VER)
      x = _byteswap_uint64(x);
#else
      x = __builtin_bswap64(x);
#endif
      runtime::memcpy(data + j, &x, sizeof(x));
    }
    return;
  case 16:
    for (std::size_t j{0}; j + 16 <= bytes; j += 16) {
      std::uint64_t hi, lo;
      runtime::memcpy(&lo, data + j, sizeof(lo));
      runtime::memcpy(&hi, data + j + 8, sizeof(hi));
#if defined(_MSC_VER)
      lo = _byteswap_uint64(lo);
      hi = _byteswap_uint64(hi);
#else
      lo = __builtin_bswap64(lo);
      hi = __builtin_bswap64(hi);
#endif
      runtime::memcpy(data + j + 8, &lo, sizeof(lo));
      runtime::memcpy(data + j, &hi, sizeof(hi));
    }
    return;
  default:;
  }
#endif
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

static constexpr std::size_t kRecMarkSize{4};
static constexpr std::int32_t kMaxSubrecPayload{0x7fffffff - 8};
// Transfers at least this large bypass the frame and are read directly.
static constexpr std::size_t kStreamWindow{65536};

static inline RT_API_ATTRS std::size_t SubrecordPayloadBytes(
    std::int32_t marker) {
  std::int64_t m{marker};
  return static_cast<std::size_t>(m < 0 ? -m : m);
}

static inline RT_API_ATTRS std::int32_t SignedRecordMarker(
    std::int32_t length, bool negative) {
  return negative ? -length : length;
}

static inline RT_API_ATTRS void StoreRecordMarker(
    char *p, std::int32_t value, bool swapEndianness) {
  runtime::memcpy(p, &value, kRecMarkSize);
  if (swapEndianness) {
    SwapEndianness(p, kRecMarkSize, kRecMarkSize);
  }
}

static inline RT_API_ATTRS std::int32_t LoadRecordMarker(
    const char *p, bool swapEndianness) {
  std::int32_t word;
  runtime::memcpy(&word, p, kRecMarkSize);
  if (swapEndianness) {
    SwapEndianness(reinterpret_cast<char *>(&word), kRecMarkSize, kRecMarkSize);
  }
  return word;
}

static inline RT_API_ATTRS std::size_t MaxSubrecordPayload() {
  std::size_t limit{static_cast<std::size_t>(kMaxSubrecPayload)};
  std::size_t configured{executionEnvironment.maxSubrecordLength};
  return configured > 0 && configured < limit ? configured : limit;
}

static RT_API_ATTRS void LayoutBufferedSubrecords(char *base,
    std::size_t payload, std::size_t maxChunk, bool swapEndianness) {
  std::size_t subrecords{(payload + maxChunk - 1) / maxChunk};
  for (std::size_t i{subrecords}; i-- > 0;) {
    std::size_t chunk{i + 1 < subrecords ? maxChunk : payload - i * maxChunk};
    std::size_t src{kRecMarkSize + i * maxChunk};
    std::size_t dst{kRecMarkSize + i * (maxChunk + 2 * kRecMarkSize)};
    if (dst != src && chunk > 0) {
      runtime::memmove(base + dst, base + src, chunk);
    }
    std::int32_t length{static_cast<std::int32_t>(chunk)};
    StoreRecordMarker(base + dst - kRecMarkSize,
        SignedRecordMarker(length, i + 1 < subrecords), swapEndianness);
    StoreRecordMarker(
        base + dst + chunk, SignedRecordMarker(length, i > 0), swapEndianness);
  }
}

void ExternalFileUnit::ResetSequentialUnformattedStream() {
  streamRead_ = false;
  streamReadRemain_ = 0;
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
    runtime::memset(Frame() + recordOffsetInFrame_ + furthestPositionInRecord,
        ' ', positionInRecord - furthestPositionInRecord);
  }
  char *to{Frame() + recordOffsetInFrame_ + positionInRecord};
  runtime::memcpy(to, data, bytes);
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
  if (streamRead_) {
    std::size_t copied{0};
    while (copied < bytes) {
      if (streamReadRemain_ == 0) {
        streamReadPos_ += kRecMarkSize;
        if (ReadFrame(streamReadPos_, kRecMarkSize, handler) < kRecMarkSize) {
          HitEndOnRead(handler);
          return false;
        }
        streamReadRemain_ =
            SubrecordPayloadBytes(LoadRecordMarker(Frame(), swapEndianness_));
        streamReadPos_ += kRecMarkSize;
      }
      std::size_t chunk{bytes - copied};
      if (chunk > streamReadRemain_) {
        chunk = streamReadRemain_;
      }
      if (chunk >= kStreamWindow) {
        if (Read(streamReadPos_, data + copied, chunk, chunk, handler) <
            chunk) {
          HitEndOnRead(handler);
          return false;
        }
      } else {
        if (ReadFrame(streamReadPos_, chunk, handler) < chunk) {
          HitEndOnRead(handler);
          return false;
        }
        runtime::memcpy(data + copied, Frame(), chunk);
      }
      streamReadPos_ += static_cast<std::int64_t>(chunk);
      streamReadRemain_ -= chunk;
      copied += chunk;
      positionInRecord += static_cast<std::int64_t>(chunk);
    }
    // Swap over the assembled buffer so that elements split across a
    // subrecord boundary are byte-reversed as a whole.
    if (swapEndianness_) {
      SwapEndianness(data, bytes, elementBytes);
    }
    furthestPositionInRecord =
        std::max(furthestPositionInRecord, positionInRecord);
    return true;
  }
  auto need{recordOffsetInFrame_ + furthestAfter};
  auto got{ReadFrame(frameOffsetInFile_, need, handler)};
  if (got >= need) {
    runtime::memcpy(
        data, Frame() + recordOffsetInFrame_ + positionInRecord, bytes);
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
  if (access == Access::Sequential &&
      positionInRecord < recordLength.value_or(positionInRecord)) {
    // Fast path for variable-length formatted input: the whole record
    // must be in frame as a result of newline detection for record length.
    p = Frame() + recordOffsetInFrame_ + positionInRecord;
    return *recordLength - positionInRecord;
  }
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
        frameOffsetInFile_ += unformattedDiskExtent_;
        unformattedDiskExtent_ = 0;
        recordOffsetInFrame_ = 0;
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
  ResetSequentialUnformattedStream();
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
        runtime::memset(
            Frame() + recordOffsetInFrame_ + furthestPositionInRecord,
            isUnformatted.value_or(false) ? 0 : ' ',
            *openRecl - furthestPositionInRecord);
        furthestPositionInRecord = *openRecl;
      }
    } else if (*isUnformatted) {
      if (access == Access::Sequential) {
        std::size_t payload{
            static_cast<std::size_t>(furthestPositionInRecord - kRecMarkSize)};
        std::size_t maxChunk{MaxSubrecordPayload()};
        if (payload <= maxChunk) {
          std::int32_t length{static_cast<std::int32_t>(payload)};
          ok = ok &&
              Emit(reinterpret_cast<const char *>(&length), kRecMarkSize,
                  kRecMarkSize, handler);
          positionInRecord = 0;
          ok = ok &&
              Emit(reinterpret_cast<const char *>(&length), kRecMarkSize,
                  kRecMarkSize, handler);
        } else {
          if (openRecl && payload > static_cast<std::size_t>(*openRecl)) {
            handler.SignalError(IostatRecordWriteOverrun,
                "Attempt to write %zd bytes to position 0 in a fixed-size "
                "record of %jd bytes",
                payload, static_cast<std::intmax_t>(*openRecl));
            return false;
          }
          std::size_t subrecords{(payload + maxChunk - 1) / maxChunk};
          std::size_t diskExtent{payload + 2 * subrecords * kRecMarkSize};
          WriteFrame(
              frameOffsetInFile_, recordOffsetInFrame_ + diskExtent, handler);
          char *base{Frame() + recordOffsetInFrame_};
          LayoutBufferedSubrecords(base, payload, maxChunk, swapEndianness_);
          furthestPositionInRecord = static_cast<std::int64_t>(diskExtent);
          positionInRecord = furthestPositionInRecord;
        }
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
    SetPosition(0);
    currentRecordNumber = 1;
    anyWriteSinceLastPositioning_ = false;
  }
}

void ExternalFileUnit::SetPosition(std::int64_t pos) {
  frameOffsetInFile_ = pos;
  recordOffsetInFrame_ = 0;
  ResetSequentialUnformattedStream();
  if (access == Access::Direct) {
    directAccessRecWasSet_ = true;
  }
  BeginRecord();
  beganReadingRecord_ = false; // for positioning after nonadvancing input
  leftTabLimit.reset();
}

void ExternalFileUnit::Sought(std::int64_t zeroBasedPos) {
  SetPosition(zeroBasedPos);
  if (zeroBasedPos == 0) {
    currentRecordNumber = 1;
  } else {
    // We no longer know which record we're in.  Set currentRecordNumber to
    // a large value from whence we can both advance and backspace.
    currentRecordNumber = common::numeric_limits<std::int64_t>::max() / 2;
    endfileRecordNumber.reset();
  }
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
  Sought(oneBasedPos - 1);
  return true;
}

// GNU FSEEK extension
RT_API_ATTRS bool ExternalFileUnit::Fseek(std::int64_t zeroBasedPos,
    enum FseekWhence whence, IoErrorHandler &handler) {
  if (whence == FseekEnd) {
    Flush(handler); // updates knownSize_
    if (auto size{knownSize()}) {
      zeroBasedPos += *size;
    } else {
      return false;
    }
  } else if (whence == FseekCurrent) {
    zeroBasedPos += InquirePos() - 1;
  }
  if (zeroBasedPos >= 0) {
    Sought(zeroBasedPos);
    return true;
  } else {
    return false;
  }
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
  SetPosition((oneBasedRec - 1) * *openRecl);
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
  unformattedDiskExtent_ = 0;
  streamRead_ = false;
  std::int64_t recordStart{
      frameOffsetInFile_ + static_cast<std::int64_t>(recordOffsetInFrame_)};
  std::size_t need{recordOffsetInFrame_ + kRecMarkSize};
  std::size_t got{ReadFrame(frameOffsetInFile_, need, handler)};
  positionInRecord = kRecMarkSize;
  if (got < need) {
    if (got <= recordOffsetInFrame_) {
      HitEndOnRead(handler);
    } else {
      handler.SignalError(
          "Unformatted variable-length sequential file input failed at "
          "record #%jd (file offset %jd): truncated record header",
          static_cast<std::intmax_t>(currentRecordNumber),
          static_cast<std::intmax_t>(recordStart));
    }
    return;
  }
  std::int32_t header{
      LoadRecordMarker(Frame() + recordOffsetInFrame_, swapEndianness_)};

  // A negative header within the subrecord-length limit begins a gfortran
  // subrecord chain.  Legacy 2–4 GiB single-marker records also have a
  // negative int32 header, but their magnitude exceeds kMaxSubrecPayload and
  // fall through to the unsigned path below.  Chain walk requires seeking.
  if (mayPosition() && header < 0 &&
      SubrecordPayloadBytes(header) <=
          static_cast<std::size_t>(kMaxSubrecPayload)) {
    auto readMarkerAt = [&](std::int64_t &pos, std::int32_t &marker) {
      char buf[kRecMarkSize];
      if (Read(pos, buf, kRecMarkSize, kRecMarkSize, handler) < kRecMarkSize) {
        return false;
      }
      pos += kRecMarkSize;
      marker = LoadRecordMarker(buf, swapEndianness_);
      return true;
    };
    std::size_t totalPayload{0};
    std::int64_t filePos{recordStart + static_cast<std::int64_t>(kRecMarkSize)};
    std::int32_t marker{header};
    bool ok{true};
    bool isFirstSubrecord{true};
    while (true) {
      totalPayload += SubrecordPayloadBytes(marker);
      filePos += static_cast<std::int64_t>(SubrecordPayloadBytes(marker));
      std::int32_t subrecordFooter{0};
      if (!readMarkerAt(filePos, subrecordFooter) ||
          SubrecordPayloadBytes(subrecordFooter) !=
              SubrecordPayloadBytes(marker) ||
          (isFirstSubrecord && subrecordFooter < 0)) {
        ok = false;
        break;
      }
      isFirstSubrecord = false;
      if (marker >= 0) {
        break;
      }
      if (!readMarkerAt(filePos, marker)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      recordLength = static_cast<std::int64_t>(kRecMarkSize + totalPayload);
      unformattedDiskExtent_ = filePos - recordStart;
      streamRead_ = true;
      streamReadPos_ = recordStart + static_cast<std::int64_t>(kRecMarkSize);
      streamReadRemain_ = SubrecordPayloadBytes(header);
      return;
    }
  }

  // Single header/footer pair.  Treat the marker as unsigned so that legacy
  // records with lengths in 2–4 GiB remain readable.
  std::uint32_t length{static_cast<std::uint32_t>(header)};
  recordLength = static_cast<std::int64_t>(kRecMarkSize + length);
  unformattedDiskExtent_ =
      static_cast<std::int64_t>(kRecMarkSize + length + kRecMarkSize);
  // For large records on seekable files peek at just the footer to avoid
  // buffering gigabytes; non-seekable files fall through to ReadFrame.
  std::uint32_t footer{0};
  bool footerOk{false};
  need = recordOffsetInFrame_ + static_cast<std::size_t>(*recordLength) +
      kRecMarkSize;
  if (length > static_cast<std::uint32_t>(kMaxSubrecPayload) && mayPosition()) {
    char footerBuf[kRecMarkSize];
    std::int64_t footerPos{
        recordStart + static_cast<std::int64_t>(kRecMarkSize + length)};
    if (Read(footerPos, footerBuf, kRecMarkSize, kRecMarkSize, handler) ==
        kRecMarkSize) {
      footer = static_cast<std::uint32_t>(
          LoadRecordMarker(footerBuf, swapEndianness_));
      footerOk = true;
    }
    got = footerOk ? need : 0;
  } else {
    got = ReadFrame(frameOffsetInFile_, need, handler);
    if (got >= need) {
      footer = static_cast<std::uint32_t>(LoadRecordMarker(
          Frame() + recordOffsetInFrame_ + *recordLength, swapEndianness_));
      footerOk = true;
    }
  }
  const char *error{nullptr};
  if (frameOffsetInFile_ == 0 && recordOffsetInFrame_ == 0 &&
      length <= static_cast<std::uint32_t>(kMaxSubrecPayload) &&
      (got < need || footer != length)) {
    // Maybe an omitted or incorrect byte swap flag setting?
    // Try it the other way, since this is the first record.
    std::uint32_t swapped{static_cast<std::uint32_t>(
        LoadRecordMarker(Frame(), !swapEndianness_))};
    std::size_t length2{kRecMarkSize + swapped};
    std::size_t need2{length2 + kRecMarkSize};
    if (ReadFrame(0, need2, handler) >= need2 &&
        static_cast<std::uint32_t>(
            LoadRecordMarker(Frame() + length2, !swapEndianness_)) == swapped) {
      error = "Unformatted variable-length sequential file input failed on the "
              "first record, probably due to a need for byte order data "
              "conversion; consider adding CONVERT='SWAP' to the OPEN "
              "statement or adding FORT_CONVERT=SWAP to the execution "
              "environment";
    }
  }
  if (error) {
  } else if (got < need) {
    error = "Unformatted variable-length sequential file input failed at "
            "record #%jd (file offset %jd): hit EOF reading record with "
            "length %jd bytes";
  } else if (footer != length) {
    error = "Unformatted variable-length sequential file input failed at "
            "record #%jd (file offset %jd): record header has length %jd "
            "that does not match record footer (%jd)";
  }
  if (error) {
    handler.SignalError(error, static_cast<std::intmax_t>(currentRecordNumber),
        static_cast<std::intmax_t>(recordStart),
        static_cast<std::intmax_t>(length), static_cast<std::intmax_t>(footer));
  }
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
  frameOffsetInFile_ += recordOffsetInFrame_;
  recordOffsetInFrame_ = 0;
  if (frameOffsetInFile_ <= static_cast<std::int64_t>(kRecMarkSize)) {
    handler.SignalError(IostatBackspaceAtFirstRecord);
    return;
  }
  if (ReadFrame(frameOffsetInFile_ - kRecMarkSize, kRecMarkSize, handler) <
      kRecMarkSize) {
    handler.SignalError(IostatShortRead);
    return;
  }
  std::int32_t trailingMarker{LoadRecordMarker(Frame(), swapEndianness_)};

  // Use the unsigned bit pattern as the step: for a legacy 2–4 GiB
  // single-marker record it equals the true payload; for a gfortran chain
  // footer it wraps past the file size, making singleMarkerStart negative.
  std::uint32_t trailingPayloadUnsigned{
      static_cast<std::uint32_t>(trailingMarker)};
  std::int64_t singleMarkerStart{frameOffsetInFile_ -
      static_cast<std::int64_t>(
          static_cast<std::uint64_t>(trailingPayloadUnsigned) +
          2 * kRecMarkSize)};

  bool isSingleMarker{false};
  if (trailingMarker >= 0) {
    isSingleMarker = true; // positive footer is always single-marker
  } else if (singleMarkerStart >= 0) {
    char hbuf[kRecMarkSize];
    if (Read(singleMarkerStart, hbuf, kRecMarkSize, kRecMarkSize, handler) ==
        kRecMarkSize) {
      isSingleMarker =
          static_cast<std::uint32_t>(LoadRecordMarker(hbuf, swapEndianness_)) ==
          static_cast<std::uint32_t>(trailingMarker);
    }
  }

  std::int32_t marker{trailingMarker};
  if (isSingleMarker) {
    if (trailingMarker >= 0) {
      std::size_t payload{SubrecordPayloadBytes(trailingMarker)};
      if (payload > static_cast<std::size_t>(kMaxSubrecPayload) ||
          frameOffsetInFile_ <
              static_cast<std::int64_t>(payload + 2 * kRecMarkSize)) {
        handler.SignalError(IostatBadUnformattedRecord);
        return;
      }
      frameOffsetInFile_ -=
          static_cast<std::int64_t>(payload + 2 * kRecMarkSize);
    } else {
      frameOffsetInFile_ = singleMarkerStart; // unsigned step, already verified
    }
  } else {
    // gfortran chain: walk backward; first subrecord's footer is non-negative.
    auto payload{SubrecordPayloadBytes(marker)};
    if (frameOffsetInFile_ <
        static_cast<std::int64_t>(payload + 2 * kRecMarkSize)) {
      handler.SignalError(IostatBadUnformattedRecord);
      return;
    }
    frameOffsetInFile_ -= static_cast<std::int64_t>(payload + 2 * kRecMarkSize);
    while (marker < 0) {
      if (ReadFrame(frameOffsetInFile_ - kRecMarkSize, kRecMarkSize, handler) <
          kRecMarkSize) {
        handler.SignalError(IostatShortRead);
        return;
      }
      marker = LoadRecordMarker(Frame(), swapEndianness_);
      payload = SubrecordPayloadBytes(marker);
      if (frameOffsetInFile_ <
          static_cast<std::int64_t>(payload + 2 * kRecMarkSize)) {
        handler.SignalError(IostatBadUnformattedRecord);
        return;
      }
      frameOffsetInFile_ -=
          static_cast<std::int64_t>(payload + 2 * kRecMarkSize);
    }
  }
  if (ReadFrame(frameOffsetInFile_, kRecMarkSize, handler) < kRecMarkSize) {
    handler.SignalError(IostatShortRead);
    return;
  }
  std::int32_t firstHeader{LoadRecordMarker(Frame(), swapEndianness_)};
  if (isSingleMarker) {
    if (static_cast<std::uint32_t>(firstHeader) !=
        static_cast<std::uint32_t>(trailingMarker)) {
      handler.SignalError(IostatBadUnformattedRecord);
    }
  } else {
    // marker is the first subrecord's footer after the backward walk.
    if (SubrecordPayloadBytes(firstHeader) != SubrecordPayloadBytes(marker)) {
      handler.SignalError(IostatBadUnformattedRecord);
    }
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
  auto lastByte{
      frameOffsetInFile_ + static_cast<std::int64_t>(recordOffsetInFrame_) - 1};
  if (lastByte < 0) {
    handler.SignalError(IostatBackspaceAtFirstRecord);
    return;
  }
  if (!(lastByte >= FrameAt() &&
          lastByte - FrameAt() < static_cast<std::int64_t>(FrameLength()))) {
    frameOffsetInFile_ = std::max<std::int64_t>(0, lastByte + 1 - 1024);
    auto need{static_cast<std::size_t>(lastByte + 1 - frameOffsetInFile_)};
    if (ReadFrame(frameOffsetInFile_, need, handler) < need) {
      handler.SignalError(IostatShortRead);
      return;
    }
  }
  bool terminated{Frame()[lastByte - FrameAt()] == '\n'};
  if (!terminated && knownSize().value_or(lastByte + 1) != lastByte + 1) {
    handler.SignalError(IostatMissingTerminator);
    return;
  }
  auto recordEnd{terminated ? lastByte : lastByte + 1};
  while (true) {
    if (frameOffsetInFile_ < recordEnd) {
      if (const char *p{
              FindLastNewline(Frame(), recordEnd - 1 - frameOffsetInFile_)}) {
        recordOffsetInFrame_ = p - Frame() + 1;
        recordLength = recordEnd - (frameOffsetInFile_ + recordOffsetInFrame_);
        break;
      }
    }
    if (frameOffsetInFile_ == 0) {
      recordOffsetInFrame_ = 0;
      recordLength = recordEnd;
      break;
    }
    frameOffsetInFile_ -= std::min<std::int64_t>(frameOffsetInFile_, 1024);
    auto need{static_cast<std::size_t>(lastByte + 1 - frameOffsetInFile_)};
    auto got{ReadFrame(frameOffsetInFile_, need, handler)};
    if (got < need) {
      handler.SignalError(IostatShortRead);
      return;
    }
  }
  if (terminated && *recordLength > 0 &&
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
  if (access != Access::Stream || executionEnvironment.truncateStream) {
    // Stream output after positioning truncates with some compilers.
    Truncate(frameOffsetInFile_, handler);
    TruncateFrame(frameOffsetInFile_, handler);
  }
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
  ChildIo *current{child_};
  Terminator &terminator{parent.GetIoErrorHandler()};
  child_ = new (AllocateMemoryOrCrash(terminator, sizeof(ChildIo)))
      ChildIo{parent, current};
  leftTabLimit = positionInRecord;
  return *child_;
}

void ExternalFileUnit::PopChildIo(ChildIo &child) {
  ChildIo *previous = child.AcquirePrevious();
  if (child_ != &child) {
    child.parent().GetIoErrorHandler().Crash(
        "ChildIo being popped is not top of stack");
  }
  child_->~ChildIo(); // delete top child
  FreeMemory(child_);
  child_ = previous;
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
