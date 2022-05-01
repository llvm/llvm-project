//===-- runtime/connection.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fortran I/O connection state (abstracted over internal & external units)

#ifndef FORTRAN_RUNTIME_IO_CONNECTION_H_
#define FORTRAN_RUNTIME_IO_CONNECTION_H_

#include "format.h"
#include <cinttypes>
#include <optional>

namespace Fortran::runtime::io {

class IoStatementState;

enum class Direction { Output, Input };
enum class Access { Sequential, Direct, Stream };

// These characteristics of a connection are immutable after being
// established in an OPEN statement.
struct ConnectionAttributes {
  Access access{Access::Sequential}; // ACCESS='SEQUENTIAL', 'DIRECT', 'STREAM'
  std::optional<bool> isUnformatted; // FORM='UNFORMATTED' if true
  bool isUTF8{false}; // ENCODING='UTF-8'
  std::optional<std::int64_t> openRecl; // RECL= on OPEN

  bool IsRecordFile() const {
    // Formatted stream files are viewed as having records, at least on input
    return access != Access::Stream || !isUnformatted.value_or(true);
  }

  template <typename CHAR = char> constexpr bool useUTF8() const {
    // For wide CHARACTER kinds, always use UTF-8 for formatted I/O.
    // For single-byte CHARACTER, encode characters >= 0x80 with
    // UTF-8 iff the mode is set.
    return sizeof(CHAR) > 1 || isUTF8;
  }
};

struct ConnectionState : public ConnectionAttributes {
  bool IsAtEOF() const; // true when read has hit EOF or endfile record
  bool IsAfterEndfile() const; // true after ENDFILE until repositioned
  std::size_t RemainingSpaceInRecord() const;
  bool NeedAdvance(std::size_t) const;
  void HandleAbsolutePosition(std::int64_t);
  void HandleRelativePosition(std::int64_t);

  void BeginRecord() {
    positionInRecord = 0;
    furthestPositionInRecord = 0;
    unterminatedRecord = false;
  }

  std::optional<std::int64_t> EffectiveRecordLength() const {
    // When an input record is longer than an explicit RECL= from OPEN
    // it is effectively truncated on input.
    return openRecl && recordLength && *openRecl < *recordLength ? openRecl
                                                                 : recordLength;
  }

  std::optional<std::int64_t> recordLength;

  // Positions in a record file (sequential or direct, not stream)
  std::int64_t currentRecordNumber{1}; // 1 is first
  std::int64_t positionInRecord{0}; // offset in current record
  std::int64_t furthestPositionInRecord{0}; // max(position+bytes)

  // Set at end of non-advancing I/O data transfer
  std::optional<std::int64_t> leftTabLimit; // offset in current record

  // currentRecordNumber value captured after ENDFILE/REWIND/BACKSPACE statement
  // or an end-of-file READ condition on a sequential access file
  std::optional<std::int64_t> endfileRecordNumber;

  // Mutable modes set at OPEN() that can be overridden in READ/WRITE & FORMAT
  MutableModes modes; // BLANK=, DECIMAL=, SIGN=, ROUND=, PAD=, DELIM=, kP

  // Set when processing repeated items during list-directed & NAMELIST input
  // in order to keep a span of records in frame on a non-positionable file,
  // so that backspacing to the beginning of the repeated item doesn't require
  // repositioning the external storage medium when that's impossible.
  bool pinnedFrame{false};

  // Set when the last record of a file is not properly terminated
  // so that a non-advancing READ will not signal EOR.
  bool unterminatedRecord{false};
};

// Utility class for capturing and restoring a position in an input stream.
class SavedPosition {
public:
  explicit SavedPosition(IoStatementState &);
  ~SavedPosition();

private:
  IoStatementState &io_;
  ConnectionState saved_;
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_CONNECTION_H_
