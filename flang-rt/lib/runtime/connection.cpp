//===-- lib/runtime/connection.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/connection.h"
#include "flang-rt/runtime/environment.h"
#include "flang-rt/runtime/io-stmt.h"

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

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

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
