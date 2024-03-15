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
#include "flang/Common/optional.h"
#include <cinttypes>

namespace Fortran::runtime::io {

class IoStatementState;

enum class Direction { Output, Input };
enum class Access { Sequential, Direct, Stream };

// These characteristics of a connection are immutable after being
// established in an OPEN statement.
struct ConnectionAttributes {
  Access access{Access::Sequential}; // ACCESS='SEQUENTIAL', 'DIRECT', 'STREAM'
  Fortran::common::optional<bool> isUnformatted; // FORM='UNFORMATTED' if true
  bool isUTF8{false}; // ENCODING='UTF-8'
  unsigned char internalIoCharKind{0}; // 0->external, 1/2/4->internal
  Fortran::common::optional<std::int64_t> openRecl; // RECL= on OPEN

  bool IsRecordFile() const {
    // Formatted stream files are viewed as having records, at least on input
    return access != Access::Stream || !isUnformatted.value_or(true);
  }

  template <typename CHAR = char> constexpr bool useUTF8() const {
    // For wide CHARACTER kinds, always use UTF-8 for formatted I/O.
    // For single-byte CHARACTER, encode characters >= 0x80 with
    // UTF-8 iff the mode is set.
    return internalIoCharKind == 0 && (sizeof(CHAR) > 1 || isUTF8);
  }
};

struct ConnectionState : public ConnectionAttributes {
  bool IsAtEOF() const; // true when read has hit EOF or endfile record
  bool IsAfterEndfile() const; // true after ENDFILE until repositioned

  // All positions and measurements are always in units of bytes,
  // not characters.  Multi-byte character encodings are possible in
  // both internal I/O (when the character kind of the variable is 2 or 4)
  // and external formatted I/O (when the encoding is UTF-8).
  std::size_t RemainingSpaceInRecord() const;
  bool NeedAdvance(std::size_t) const;
  void HandleAbsolutePosition(std::int64_t);
  void HandleRelativePosition(std::int64_t);

  void BeginRecord() {
    positionInRecord = 0;
    furthestPositionInRecord = 0;
    unterminatedRecord = false;
  }

  Fortran::common::optional<std::int64_t> EffectiveRecordLength() const {
    // When an input record is longer than an explicit RECL= from OPEN
    // it is effectively truncated on input.
    return openRecl && recordLength && *openRecl < *recordLength ? openRecl
                                                                 : recordLength;
  }

  Fortran::common::optional<std::int64_t> recordLength;

  std::int64_t currentRecordNumber{1}; // 1 is first

  // positionInRecord is the 0-based bytes offset in the current record
  // to/from which the next data transfer will occur.  It can be past
  // furthestPositionInRecord if moved by an X or T or TR control edit
  // descriptor.
  std::int64_t positionInRecord{0};

  // furthestPositionInRecord is the 0-based byte offset of the greatest
  // position in the current record to/from which any data transfer has
  // occurred, plus one.  It can be viewed as a count of bytes processed.
  std::int64_t furthestPositionInRecord{0}; // max(position+bytes)

  // Set at end of non-advancing I/O data transfer
  Fortran::common::optional<std::int64_t>
      leftTabLimit; // offset in current record

  // currentRecordNumber value captured after ENDFILE/REWIND/BACKSPACE statement
  // or an end-of-file READ condition on a sequential access file
  Fortran::common::optional<std::int64_t> endfileRecordNumber;

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
  void Cancel() { cancelled_ = true; }

private:
  IoStatementState &io_;
  ConnectionState saved_;
  bool cancelled_{false};
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_CONNECTION_H_
