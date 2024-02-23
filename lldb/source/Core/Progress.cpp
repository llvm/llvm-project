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

#include <mutex>
#include <optional>

using namespace lldb;
using namespace lldb_private;

std::atomic<uint64_t> Progress::g_id(0);

Progress::Progress(std::string title, std::string details,
                   std::optional<uint64_t> total,
                   lldb_private::Debugger *debugger)
    : m_title(title), m_details(details), m_id(++g_id), m_completed(0),
      m_total(Progress::kNonDeterministicTotal) {
  if (total)
    m_total = *total;

  if (debugger)
    m_debugger_id = debugger->GetID();
  std::lock_guard<std::mutex> guard(m_mutex);
  ReportProgress();
}

Progress::~Progress() {
  // Make sure to always report progress completed when this object is
  // destructed so it indicates the progress dialog/activity should go away.
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_completed)
    m_completed = m_total;
  ReportProgress();
}

void Progress::Increment(uint64_t amount,
                         std::optional<std::string> updated_detail) {
  if (amount > 0) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (updated_detail)
      m_details = std::move(updated_detail.value());
    // Watch out for unsigned overflow and make sure we don't increment too
    // much and exceed m_total.
    if (m_total && (amount > (m_total - m_completed)))
      m_completed = m_total;
    else
      m_completed += amount;
    ReportProgress();
  }
}

void Progress::ReportProgress() {
  if (!m_complete) {
    // Make sure we only send one notification that indicates the progress is
    // complete
    m_complete = m_completed == m_total;
    Debugger::ReportProgress(m_id, m_title, m_details, m_completed, m_total,
                             m_debugger_id);
  }
}

ProgressManager::ProgressManager() : m_progress_category_map() {}

ProgressManager::~ProgressManager() {}

ProgressManager &ProgressManager::Instance() {
  static std::once_flag g_once_flag;
  static ProgressManager *g_progress_manager = nullptr;
  std::call_once(g_once_flag, []() {
    // NOTE: known leak to avoid global destructor chain issues.
    g_progress_manager = new ProgressManager();
  });
  return *g_progress_manager;
}

void ProgressManager::Increment(std::string title) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  m_progress_category_map[title]++;
}

void ProgressManager::Decrement(std::string title) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  auto pos = m_progress_category_map.find(title);

  if (pos == m_progress_category_map.end())
    return;

  if (pos->second <= 1)
    m_progress_category_map.erase(title);
  else
    --pos->second;
}
