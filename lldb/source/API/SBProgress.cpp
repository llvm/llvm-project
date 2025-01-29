//===-- SBProgress.cpp --------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProgress.h"
#include "lldb/Core/Progress.h"
#include "lldb/Utility/Instrumentation.h"

using namespace lldb;

SBProgress::SBProgress(const char *title, const char *details,
                       SBDebugger &debugger) {
  LLDB_INSTRUMENT_VA(this, title, details, debugger);

  m_opaque_up = std::make_unique<lldb_private::Progress>(
      title, details, /*total=*/std::nullopt, debugger.get(),
      /*minimum_report_time=*/std::nullopt,
      lldb_private::Progress::Origin::eExternal);
}

SBProgress::SBProgress(const char *title, const char *details,
                       uint64_t total_units, SBDebugger &debugger) {
  LLDB_INSTRUMENT_VA(this, title, details, total_units, debugger);

  m_opaque_up = std::make_unique<lldb_private::Progress>(
      title, details, total_units, debugger.get(),
      /*minimum_report_time=*/std::nullopt,
      lldb_private::Progress::Origin::eExternal);
}

SBProgress::SBProgress(SBProgress &&rhs)
    : m_opaque_up(std::move(rhs.m_opaque_up)) {}

SBProgress::~SBProgress() = default;

void SBProgress::Increment(uint64_t amount, const char *description) {
  LLDB_INSTRUMENT_VA(amount, description);

  m_opaque_up->Increment(amount, description);
}

lldb_private::Progress &SBProgress::ref() const { return *m_opaque_up; }
