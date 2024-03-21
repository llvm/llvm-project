//===-- SBStatisticsOptions.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStatisticsOptions.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Utility/Instrumentation.h"

#include "Utils.h"

using namespace lldb;
using namespace lldb_private;

SBStatisticsOptions::SBStatisticsOptions()
    : m_opaque_up(new StatisticsOptions()) {
  LLDB_INSTRUMENT_VA(this);
  m_opaque_up->summary_only = false;
}

SBStatisticsOptions::SBStatisticsOptions(const SBStatisticsOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  m_opaque_up = clone(rhs.m_opaque_up);
}

SBStatisticsOptions::~SBStatisticsOptions() = default;

const SBStatisticsOptions &
SBStatisticsOptions::operator=(const SBStatisticsOptions &rhs) {
  LLDB_INSTRUMENT_VA(this, rhs);

  if (this != &rhs)
    m_opaque_up = clone(rhs.m_opaque_up);
  return *this;
}

void SBStatisticsOptions::SetSummaryOnly(bool b) {
  m_opaque_up->summary_only = b;
}

bool SBStatisticsOptions::GetSummaryOnly() { return m_opaque_up->summary_only; }

void SBStatisticsOptions::SetReportAllAvailableDebugInfo(bool b) {
  m_opaque_up->load_all_debug_info = b;
}

bool SBStatisticsOptions::GetReportAllAvailableDebugInfo() {
  return m_opaque_up->load_all_debug_info;
}

const lldb_private::StatisticsOptions &SBStatisticsOptions::ref() const {
  return *m_opaque_up;
}
