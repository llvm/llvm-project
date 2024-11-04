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

#include <cstdint>
#include <mutex>
#include <optional>

using namespace lldb;
using namespace lldb_private;

std::atomic<uint64_t> Progress::g_id(0);

Progress::Progress(std::string title, std::string details,
                   std::optional<uint64_t> total,
                   lldb_private::Debugger *debugger)
    : m_details(details), m_completed(0),
      m_total(Progress::kNonDeterministicTotal),
      m_progress_data{title, ++g_id,
                      /*m_progress_data.debugger_id=*/std::nullopt} {
  if (total)
    m_total = *total;

  if (debugger)
    m_progress_data.debugger_id = debugger->GetID();

  std::lock_guard<std::mutex> guard(m_mutex);
  ReportProgress();
  ProgressManager::Instance().Increment(m_progress_data);
}

Progress::~Progress() {
  // Make sure to always report progress completed when this object is
  // destructed so it indicates the progress dialog/activity should go away.
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_completed)
    m_completed = m_total;
  ReportProgress();
  ProgressManager::Instance().Decrement(m_progress_data);
}

void Progress::Increment(uint64_t amount,
                         std::optional<std::string> updated_detail) {
  if (amount > 0) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (updated_detail)
      m_details = std::move(updated_detail.value());
    // Watch out for unsigned overflow and make sure we don't increment too
    // much and exceed the total.
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
    Debugger::ReportProgress(m_progress_data.progress_id, m_progress_data.title,
                             m_details, m_completed, m_total,
                             m_progress_data.debugger_id);
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

void ProgressManager::Increment(const Progress::ProgressData &progress_data) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  // If the current category exists in the map then it is not an initial report,
  // therefore don't broadcast to the category bit. Also, store the current
  // progress data in the map so that we have a note of the ID used for the
  // initial progress report.
  if (!m_progress_category_map.contains(progress_data.title)) {
    m_progress_category_map[progress_data.title].second = progress_data;
    ReportProgress(progress_data, EventType::Begin);
  }
  m_progress_category_map[progress_data.title].first++;
}

void ProgressManager::Decrement(const Progress::ProgressData &progress_data) {
  std::lock_guard<std::mutex> lock(m_progress_map_mutex);
  auto pos = m_progress_category_map.find(progress_data.title);

  if (pos == m_progress_category_map.end())
    return;

  if (pos->second.first <= 1) {
    ReportProgress(pos->second.second, EventType::End);
    m_progress_category_map.erase(progress_data.title);
  } else {
    --pos->second.first;
  }
}

void ProgressManager::ReportProgress(
    const Progress::ProgressData &progress_data, EventType type) {
  // The category bit only keeps track of when progress report categories have
  // started and ended, so clear the details and reset other fields when
  // broadcasting to it since that bit doesn't need that information.
  const uint64_t completed =
      (type == EventType::Begin) ? 0 : Progress::kNonDeterministicTotal;
  Debugger::ReportProgress(progress_data.progress_id, progress_data.title, "",
                           completed, Progress::kNonDeterministicTotal,
                           progress_data.debugger_id,
                           Debugger::eBroadcastBitProgressCategory);
}
