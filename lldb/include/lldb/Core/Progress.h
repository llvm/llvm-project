//===-- Progress.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_PROGRESS_H
#define LLDB_CORE_PROGRESS_H

#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-types.h"
#include <atomic>
#include <mutex>
#include <optional>

namespace lldb_private {

/// A Progress indicator helper class.
///
/// Any potentially long running sections of code in LLDB should report
/// progress so that clients are aware of delays that might appear during
/// debugging. Delays commonly include indexing debug information, parsing
/// symbol tables for object files, downloading symbols from remote
/// repositories, and many more things.
///
/// The Progress class helps make sure that progress is correctly reported
/// and will always send an initial progress update, updates when
/// Progress::Increment() is called, and also will make sure that a progress
/// completed update is reported even if the user doesn't explicitly cause one
/// to be sent.
///
/// The progress is reported via a callback whose type is ProgressCallback:
///
///   typedef void (*ProgressCallback)(uint64_t progress_id,
///                                    const char *message,
///                                    uint64_t completed,
///                                    uint64_t total,
///                                    void *baton);
///
/// This callback will always initially be called with "completed" set to zero
/// and "total" set to the total amount specified in the contructor. This is
/// considered the progress start event. As Progress::Increment() is called,
/// the callback will be called as long as the Progress::m_completed has not
/// yet exceeded the Progress::m_total. When the callback is called with
/// Progress::m_completed == Progress::m_total, that is considered a progress
/// completed event. If Progress::m_completed is non-zero and less than
/// Progress::m_total, then this is considered a progress update event.
///
/// This callback will be called in the destructor if Progress::m_completed is
/// not equal to Progress::m_total with the "completed" set to
/// Progress::m_total. This ensures we always send a progress completed update
/// even if the user does not.

class Progress {
public:
  /// Construct a progress object that will report information.
  ///
  /// The constructor will create a unique progress reporting object and
  /// immediately send out a progress update by calling the installed callback
  /// with completed set to zero out of the specified total.
  ///
  /// @param [in] title The title of this progress activity.
  ///
  /// @param [in] total The total units of work to be done if specified, if
  /// set to std::nullopt then an indeterminate progress indicator should be
  /// displayed.
  ///
  /// @param [in] debugger An optional debugger pointer to specify that this
  /// progress is to be reported only to specific debuggers.
  Progress(std::string title, std::string details = {},
           std::optional<uint64_t> total = std::nullopt,
           lldb_private::Debugger *debugger = nullptr);

  /// Destroy the progress object.
  ///
  /// If the progress has not yet sent a completion update, the destructor
  /// will send out a notification where the completed == m_total. This ensures
  /// that we always send out a progress complete notification.
  ~Progress();

  /// Increment the progress and send a notification to the intalled callback.
  ///
  /// If incrementing ends up exceeding m_total, m_completed will be updated
  /// to match m_total and no subsequent progress notifications will be sent.
  /// If no total was specified in the constructor, this function will not do
  /// anything nor send any progress updates.
  ///
  /// @param [in] amount The amount to increment m_completed by.
  ///
  /// @param [in] an optional message associated with this update.
  void Increment(uint64_t amount = 1,
                 std::optional<std::string> updated_detail = {});

  /// Used to indicate a non-deterministic progress report
  static constexpr uint64_t kNonDeterministicTotal = UINT64_MAX;

private:
  void ReportProgress();
  static std::atomic<uint64_t> g_id;
  /// The title of the progress activity.
  std::string m_title;
  std::string m_details;
  std::mutex m_mutex;
  /// A unique integer identifier for progress reporting.
  const uint64_t m_id;
  /// How much work ([0...m_total]) that has been completed.
  uint64_t m_completed;
  /// Total amount of work, use a std::nullopt in the constructor for non
  /// deterministic progress.
  uint64_t m_total;
  /// The optional debugger ID to report progress to. If this has no value then
  /// all debuggers will receive this event.
  std::optional<lldb::user_id_t> m_debugger_id;
  /// Set to true when progress has been reported where m_completed == m_total
  /// to ensure that we don't send progress updates after progress has
  /// completed.
  bool m_complete = false;
};

} // namespace lldb_private

#endif // LLDB_CORE_PROGRESS_H
