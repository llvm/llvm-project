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

enum class Direction { Output, Input };
enum class Access { Sequential, Direct, Stream };

inline bool IsRecordFile(Access a) { return a != Access::Stream; }

// These characteristics of a connection are immutable after being
// established in an OPEN statement.
struct ConnectionAttributes {
  Access access{Access::Sequential}; // ACCESS='SEQUENTIAL', 'DIRECT', 'STREAM'
  std::optional<bool> isUnformatted; // FORM='UNFORMATTED' if true
  bool isUTF8{false}; // ENCODING='UTF-8'
  bool isFixedRecordLength{false}; // RECL= on OPEN
  std::optional<std::int64_t> recordLength; // RECL= or current record
};

struct ConnectionState : public ConnectionAttributes {
  bool IsAtEOF() const; // true when read has hit EOF or endfile record
  std::size_t RemainingSpaceInRecord() const;
  bool NeedAdvance(std::size_t) const;
  void HandleAbsolutePosition(std::int64_t);
  void HandleRelativePosition(std::int64_t);

  void BeginRecord() {
    positionInRecord = 0;
    furthestPositionInRecord = 0;
    leftTabLimit.reset();
  }

  // Positions in a record file (sequential or direct, not stream)
  std::int64_t currentRecordNumber{1}; // 1 is first
  std::int64_t positionInRecord{0}; // offset in current record
  std::int64_t furthestPositionInRecord{0}; // max(position+bytes)
  bool nonAdvancing{false}; // ADVANCE='NO'

  // Set at end of non-advancing I/O data transfer
  std::optional<std::int64_t> leftTabLimit; // offset in current record

  // currentRecordNumber value captured after ENDFILE/REWIND/BACKSPACE statement
  // or an end-of-file READ condition on a sequential access file
  std::optional<std::int64_t> endfileRecordNumber;

  // Mutable modes set at OPEN() that can be overridden in READ/WRITE & FORMAT
  MutableModes modes; // BLANK=, DECIMAL=, SIGN=, ROUND=, PAD=, DELIM=, kP
};
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_CONNECTION_H_
