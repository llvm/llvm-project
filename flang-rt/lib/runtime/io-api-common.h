//===-- lib/runtime/io-api-common.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_IO_API_COMMON_H_
#define FLANG_RT_RUNTIME_IO_API_COMMON_H_

#include "unit.h"
#include "flang-rt/runtime/io-stmt.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Common/api-attrs.h"
#include "flang/Common/optional.h"
#include "flang/Runtime/io-api.h"

namespace Fortran::runtime::io {

static inline RT_API_ATTRS Cookie NoopUnit(const Terminator &terminator,
    int unitNumber, enum Iostat iostat = IostatOk) {
  Cookie cookie{&New<NoopStatementState>{terminator}(
      terminator.sourceFileName(), terminator.sourceLine(), unitNumber)
                     .release()
                     ->ioStatementState()};
  if (iostat != IostatOk) {
    cookie->GetIoErrorHandler().SetPendingError(iostat);
  }
  return cookie;
}

static inline RT_API_ATTRS ExternalFileUnit *GetOrCreateUnit(int unitNumber,
    Direction direction, common::optional<bool> isUnformatted,
    const Terminator &terminator, Cookie &errorCookie) {
  IoErrorHandler handler{terminator};
  handler.HasIoStat();
  if (ExternalFileUnit *
      unit{ExternalFileUnit::LookUpOrCreateAnonymous(
          unitNumber, direction, isUnformatted, handler)}) {
    errorCookie = nullptr;
    return unit;
  } else {
    auto iostat{static_cast<enum Iostat>(handler.GetIoStat())};
    errorCookie = NoopUnit(terminator, unitNumber,
        iostat != IostatOk ? iostat : IostatBadUnitNumber);
    return nullptr;
  }
}

template <Direction DIR, template <Direction> class STATE, typename... A>
RT_API_ATTRS Cookie BeginExternalListIO(
    int unitNumber, const char *sourceFile, int sourceLine, A &&...xs) {
  Terminator terminator{sourceFile, sourceLine};
  Cookie errorCookie{nullptr};
  ExternalFileUnit *unit{GetOrCreateUnit(
      unitNumber, DIR, false /*!unformatted*/, terminator, errorCookie)};
  if (!unit) {
    return errorCookie;
  }
  if (!unit->isUnformatted.has_value()) {
    unit->isUnformatted = false;
  }
  Iostat iostat{IostatOk};
  if (*unit->isUnformatted) {
    iostat = IostatFormattedIoOnUnformattedUnit;
  }
  if (ChildIo * child{unit->GetChildIo()}) {
    if (iostat == IostatOk) {
      iostat = child->CheckFormattingAndDirection(false, DIR);
    }
    if (iostat == IostatOk) {
      return &child->BeginIoStatement<ChildListIoStatementState<DIR>>(
          *child, sourceFile, sourceLine);
    } else {
      return &child->BeginIoStatement<ErroneousIoStatementState>(
          iostat, nullptr /* no unit */, sourceFile, sourceLine);
    }
  } else {
    if (iostat == IostatOk && unit->access == Access::Direct) {
      iostat = IostatListIoOnDirectAccessUnit;
    }
    if (iostat == IostatOk) {
      iostat = unit->SetDirection(DIR);
    }
    if (iostat == IostatOk) {
      return &unit->BeginIoStatement<STATE<DIR>>(
          terminator, std::forward<A>(xs)..., *unit, sourceFile, sourceLine);
    } else {
      return &unit->BeginIoStatement<ErroneousIoStatementState>(
          terminator, iostat, unit, sourceFile, sourceLine);
    }
  }
}

} // namespace Fortran::runtime::io
#endif // FLANG_RT_RUNTIME_IO_API_COMMON_H_
