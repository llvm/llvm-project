//===-- Progress.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Progress.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb;
using namespace lldb_private;

std::atomic<uint64_t> Progress::g_id(0);

Progress::Progress(std::string title, uint64_t total,
                   lldb_private::Debugger *debugger)
    : m_title(title), m_id(++g_id), m_completed(0), m_total(total) {
  assert(total > 0);
  if (debugger)
    m_debugger_id = debugger->GetID();
  std::lock_guard<std::mutex> guard(m_mutex);
  ReportProgress();
}

Progress::~Progress() {
  // Make sure to always report progress completed when this object is
  // destructed so it indicates the progress dialog/activity should go away.
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_completed) {
    m_completed = m_total;
    ReportProgress();
  }
}

void Progress::Increment(uint64_t amount, std::string update) {
  if (amount > 0) {
    std::lock_guard<std::mutex> guard(m_mutex);
    // Watch out for unsigned overflow and make sure we don't increment too
    // much and exceed m_total.
    if (amount > (m_total - m_completed))
      m_completed = m_total;
    else
      m_completed += amount;
    ReportProgress(update);
  }
}

void Progress::ReportProgress(std::string update) {
  if (!m_complete) {
    // Make sure we only send one notification that indicates the progress is
    // complete.
    m_complete = m_completed == m_total;
    Debugger::ReportProgress(m_id, m_title, std::move(update), m_completed,
                             m_total, m_debugger_id);
  }
}
