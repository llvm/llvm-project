//===-- runtime/connection.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "connection.h"
#include "environment.h"
#include "io-stmt.h"
#include <algorithm>

namespace Fortran::runtime::io {

std::size_t ConnectionState::RemainingSpaceInRecord() const {
  auto recl{recordLength.value_or(openRecl.value_or(
      executionEnvironment.listDirectedOutputLineLengthLimit))};
  return positionInRecord >= recl ? 0 : recl - positionInRecord;
}

bool ConnectionState::NeedAdvance(std::size_t width) const {
  return positionInRecord > 0 && width > RemainingSpaceInRecord();
}

bool ConnectionState::IsAtEOF() const {
  return endfileRecordNumber && currentRecordNumber >= *endfileRecordNumber;
}

bool ConnectionState::IsAfterEndfile() const {
  return endfileRecordNumber && currentRecordNumber > *endfileRecordNumber;
}

void ConnectionState::HandleAbsolutePosition(std::int64_t n) {
  positionInRecord = std::max(n, std::int64_t{0}) + leftTabLimit.value_or(0);
}

void ConnectionState::HandleRelativePosition(std::int64_t n) {
  positionInRecord = std::max(leftTabLimit.value_or(0), positionInRecord + n);
}

SavedPosition::SavedPosition(IoStatementState &io) : io_{io} {
  ConnectionState &conn{io_.GetConnectionState()};
  saved_ = conn;
  conn.pinnedFrame = true;
}

SavedPosition::~SavedPosition() {
  if (!cancelled_) {
    ConnectionState &conn{io_.GetConnectionState()};
    while (conn.currentRecordNumber > saved_.currentRecordNumber) {
      io_.BackspaceRecord();
    }
    conn.leftTabLimit = saved_.leftTabLimit;
    conn.furthestPositionInRecord = saved_.furthestPositionInRecord;
    conn.positionInRecord = saved_.positionInRecord;
    conn.pinnedFrame = saved_.pinnedFrame;
  }
}
} // namespace Fortran::runtime::io
