//===-- runtime/connection.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "connection.h"
#include "environment.h"
#include <algorithm>

namespace Fortran::runtime::io {

std::size_t ConnectionState::RemainingSpaceInRecord() const {
  auto recl{recordLength.value_or(
      executionEnvironment.listDirectedOutputLineLengthLimit)};
  return positionInRecord >= recl ? 0 : recl - positionInRecord;
}

bool ConnectionState::IsAtEOF() const {
  return endfileRecordNumber && currentRecordNumber >= *endfileRecordNumber;
}

void ConnectionState::HandleAbsolutePosition(std::int64_t n) {
  positionInRecord = std::max(n, std::int64_t{0}) + leftTabLimit.value_or(0);
}

void ConnectionState::HandleRelativePosition(std::int64_t n) {
  positionInRecord = std::max(leftTabLimit.value_or(0), positionInRecord + n);
}
} // namespace Fortran::runtime::io
