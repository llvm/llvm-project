//===-- Progress.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Progress.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Utility/LLDBLog.h"
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

  // Using a shared_future because std::function needs to be copyable.
  if (llvm::Expected<HostThread> reporting_thread =
          ThreadLauncher::LaunchThread(
              "<lldb.progress>",
              [this, future = std::shared_future<void>(
                         m_stop_reporting_thread.get_future())]() {
                SendPeriodicReports(future);
                return lldb::thread_result_t();
              })) {
    m_reporting_thread = std::move(*reporting_thread);
  } else {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Host), reporting_thread.takeError(),
                   "failed to launch host thread: {}");
  }
}

Progress::~Progress() {
  m_stop_reporting_thread.set_value();
  if (m_reporting_thread.IsJoinable()) {
    m_reporting_thread.Join(nullptr);
  }
}

void Progress::SendPeriodicReports(std::shared_future<void> done) {
  uint64_t last_completed = 0;
  Debugger::ReportProgress(m_id, m_title, "", last_completed, m_total,
                           m_debugger_id);

  while (last_completed != m_total &&
         done.wait_for(std::chrono::milliseconds(100)) ==
             std::future_status::timeout) {
    uint64_t current_completed = m_completed.load();
    if (current_completed == last_completed)
      continue;

    if (current_completed == m_total ||
        current_completed < last_completed /*overflow*/) {
      break;
    }

    std::string current_update;
    {
      std::lock_guard<std::mutex> guard(m_update_mutex);
      current_update = std::move(m_update);
      m_update.clear();
    }
    Debugger::ReportProgress(m_id, m_title, std::move(current_update),
                             current_completed, m_total, m_debugger_id);
    last_completed = current_completed;
  }

  Debugger::ReportProgress(m_id, m_title, "", m_total, m_total, m_debugger_id);
}

void Progress::Increment(uint64_t amount, std::string update) {
  if (amount == 0)
    return;
  if (!update.empty()) {
    std::lock_guard<std::mutex> guard(m_update_mutex);
    m_update = std::move(update);
  }
  m_completed += amount;
}
