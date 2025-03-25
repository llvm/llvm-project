//===-- Progress.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_PROGRESS_H
#define LLDB_CORE_PROGRESS_H

#include "lldb/Utility/Timeout.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/StringMap.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

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
/// This callback will always initially be called with \a completed set to zero
/// and \a total set to the total amount specified in the constructor. This is
/// considered the progress start event. As Progress::Increment() is called,
/// the callback will be called as long as the Progress::m_completed has not
/// yet exceeded the Progress::m_total. When the callback is called with
/// Progress::m_completed == Progress::m_total, that is considered a progress
/// completed event. If Progress::m_completed is non-zero and less than
/// Progress::m_total, then this is considered a progress update event.
///
/// This callback will be called in the destructor if Progress::m_completed is
/// not equal to Progress::m_total with the \a completed set to
/// Progress::m_total. This ensures we always send a progress completed update
/// even if the user does not.

class Progress {
public:
  /// Enum to indicate the origin of a progress event, internal or external.
  enum class Origin : uint8_t {
    eInternal = 0,
    eExternal = 1,
  };

  /// Construct a progress object that will report information.
  ///
  /// The constructor will create a unique progress reporting object and
  /// immediately send out a progress update by calling the installed callback
  /// with \a completed set to zero out of the specified total.
  ///
  /// @param [in] title The title of this progress activity.
  ///
  /// @param [in] details Specific information about what the progress report
  /// is currently working on. Although not required, if the progress report is
  /// updated with Progress::Increment() then this field will be overwritten
  /// with the new set of details passed into that function, and the details
  /// passed initially will act as an "item 0" for the total set of
  /// items being reported on.
  ///
  /// @param [in] total The total units of work to be done if specified, if
  /// set to std::nullopt then an indeterminate progress indicator should be
  /// displayed.
  ///
  /// @param [in] debugger An optional debugger pointer to specify that this
  /// progress is to be reported only to specific debuggers.
  Progress(std::string title, std::string details = {},
           std::optional<uint64_t> total = std::nullopt,
           lldb_private::Debugger *debugger = nullptr,
           Timeout<std::nano> minimum_report_time = std::nullopt,
           Origin origin = Origin::eInternal);

  /// Destroy the progress object.
  ///
  /// If the progress has not yet sent a completion update, the destructor
  /// will send out a notification where the \a completed == m_total. This
  /// ensures that we always send out a progress complete notification.
  ~Progress();

  /// Increment the progress and send a notification to the installed callback.
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

  /// Total amount of work, use a std::nullopt in the constructor for non
  /// deterministic progress.
  const uint64_t m_total;

  // Minimum amount of time between two progress reports.
  const Timeout<std::nano> m_minimum_report_time;

  /// The title of the progress activity, also used as a category.
  const std::string m_title;

  /// A unique integer identifier for progress reporting.
  const uint64_t m_progress_id;

  /// The optional debugger ID to report progress to. If this has no value
  /// then all debuggers will receive this event.
  const std::optional<lldb::user_id_t> m_debugger_id;

  /// The origin of the progress event, whether it is internal or external.
  const Origin m_origin;

  /// How much work ([0...m_total]) that has been completed.
  std::atomic<uint64_t> m_completed = 0;

  /// Time (in nanoseconds since epoch) of the last progress report.
  std::atomic<uint64_t> m_last_report_time_ns;

  /// Guards non-const non-atomic members of the class.
  std::mutex m_mutex;

  /// More specific information about the current file being displayed in the
  /// report.
  std::string m_details;

  /// The "completed" value of the last reported event.
  std::optional<uint64_t> m_prev_completed;
};

} // namespace lldb_private

#endif // LLDB_CORE_PROGRESS_H
