//===-- Statistics.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_STATISTICS_H
#define LLDB_TARGET_STATISTICS_H

#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/RealpathPrefixes.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <vector>

namespace lldb_private {

using StatsClock = std::chrono::high_resolution_clock;
using StatsTimepoint = std::chrono::time_point<StatsClock>;
class SummaryStatistics;
// Declaring here as there is no private forward
typedef std::shared_ptr<SummaryStatistics> SummaryStatisticsSP;

class StatsDuration {
public:
  using Duration = std::chrono::duration<double>;

  Duration get() const {
    return Duration(InternalDuration(value.load(std::memory_order_relaxed)));
  }
  operator Duration() const { return get(); }

  void reset() { value.store(0, std::memory_order_relaxed); }

  StatsDuration &operator+=(Duration dur) {
    value.fetch_add(std::chrono::duration_cast<InternalDuration>(dur).count(),
                    std::memory_order_relaxed);
    return *this;
  }

private:
  using InternalDuration = std::chrono::duration<uint64_t, std::micro>;
  std::atomic<uint64_t> value{0};
};

/// A class that measures elapsed time in an exception safe way.
///
/// This is a RAII class is designed to help gather timing statistics within
/// LLDB where objects have optional Duration variables that get updated with
/// elapsed times. This helps LLDB measure statistics for many things that are
/// then reported in LLDB commands.
///
/// Objects that need to measure elapsed times should have a variable of type
/// "StatsDuration m_time_xxx;" which can then be used in the constructor of
/// this class inside a scope that wants to measure something:
///
///   ElapsedTime elapsed(m_time_xxx);
///   // Do some work
///
/// This class will increment the m_time_xxx variable with the elapsed time
/// when the object goes out of scope. The "m_time_xxx" variable will be
/// incremented when the class goes out of scope. This allows a variable to
/// measure something that might happen in stages at different times, like
/// resolving a breakpoint each time a new shared library is loaded.
class ElapsedTime {
public:
  /// Set to the start time when the object is created.
  StatsTimepoint m_start_time;
  /// Elapsed time in seconds to increment when this object goes out of scope.
  StatsDuration &m_elapsed_time;

public:
  ElapsedTime(StatsDuration &opt_time) : m_elapsed_time(opt_time) {
    m_start_time = StatsClock::now();
  }
  ~ElapsedTime() {
    StatsClock::duration elapsed = StatsClock::now() - m_start_time;
    m_elapsed_time += elapsed;
  }
};

/// A class to count success/fail statistics.
struct StatsSuccessFail {
  StatsSuccessFail(llvm::StringRef n) : name(n.str()) {}

  void NotifySuccess() { ++successes; }
  void NotifyFailure() { ++failures; }

  llvm::json::Value ToJSON() const;
  std::string name;
  uint32_t successes = 0;
  uint32_t failures = 0;
};

/// A class that represents statistics for a since lldb_private::Module.
struct ModuleStats {
  llvm::json::Value ToJSON() const;
  intptr_t identifier;
  std::string path;
  std::string uuid;
  std::string triple;
  // Path separate debug info file, or empty if none.
  std::string symfile_path;
  // If the debug info is contained in multiple files where each one is
  // represented as a separate lldb_private::Module, then these are the
  // identifiers of these modules in the global module list. This allows us to
  // track down all of the stats that contribute to this module.
  std::vector<intptr_t> symfile_modules;
  llvm::StringMap<llvm::json::Value> type_system_stats;
  double symtab_parse_time = 0.0;
  double symtab_index_time = 0.0;
  double debug_parse_time = 0.0;
  double debug_index_time = 0.0;
  uint64_t debug_info_size = 0;
  bool symtab_loaded_from_cache = false;
  bool symtab_saved_to_cache = false;
  bool debug_info_index_loaded_from_cache = false;
  bool debug_info_index_saved_to_cache = false;
  bool debug_info_enabled = true;
  bool symtab_stripped = false;
  bool debug_info_had_variable_errors = false;
  bool debug_info_had_incomplete_types = false;
};

struct ConstStringStats {
  llvm::json::Value ToJSON() const;
  ConstString::MemoryStats stats = ConstString::GetMemoryStats();
};

struct StatisticsOptions {
public:
  void SetSummaryOnly(bool value) { m_summary_only = value; }
  bool GetSummaryOnly() const { return m_summary_only.value_or(false); }

  void SetLoadAllDebugInfo(bool value) { m_load_all_debug_info = value; }
  bool GetLoadAllDebugInfo() const {
    return m_load_all_debug_info.value_or(false);
  }

  void SetIncludeTargets(bool value) { m_include_targets = value; }
  bool GetIncludeTargets() const {
    if (m_include_targets.has_value())
      return m_include_targets.value();
    // Default to true in both default mode and summary mode.
    return true;
  }

  void SetIncludeModules(bool value) { m_include_modules = value; }
  bool GetIncludeModules() const {
    if (m_include_modules.has_value())
      return m_include_modules.value();
    // `m_include_modules` has no value set, so return a value based on
    // `m_summary_only`.
    return !GetSummaryOnly();
  }

  void SetIncludeTranscript(bool value) { m_include_transcript = value; }
  bool GetIncludeTranscript() const {
    if (m_include_transcript.has_value())
      return m_include_transcript.value();
    // `m_include_transcript` has no value set, so return a value based on
    // `m_summary_only`.
    return !GetSummaryOnly();
  }

private:
  std::optional<bool> m_summary_only;
  std::optional<bool> m_load_all_debug_info;
  std::optional<bool> m_include_targets;
  std::optional<bool> m_include_modules;
  std::optional<bool> m_include_transcript;
};

/// A class that represents statistics about a TypeSummaryProviders invocations
/// \note All members of this class need to be accessed in a thread safe manner
class SummaryStatistics {
public:
  explicit SummaryStatistics(std::string name, std::string impl_type)
      : m_total_time(), m_impl_type(std::move(impl_type)),
        m_name(std::move(name)), m_count(0) {}

  std::string GetName() const { return m_name; };
  double GetTotalTime() const { return m_total_time.get().count(); }

  uint64_t GetSummaryCount() const {
    return m_count.load(std::memory_order_relaxed);
  }

  StatsDuration &GetDurationReference() { return m_total_time; };

  std::string GetSummaryKindName() const { return m_impl_type; }

  llvm::json::Value ToJSON() const;

  void Reset() { m_total_time.reset(); }

  /// Basic RAII class to increment the summary count when the call is complete.
  class SummaryInvocation {
  public:
    SummaryInvocation(SummaryStatisticsSP summary_stats)
        : m_stats(summary_stats),
          m_elapsed_time(summary_stats->GetDurationReference()) {}
    ~SummaryInvocation() { m_stats->OnInvoked(); }

    /// Delete the copy constructor and assignment operator to prevent
    /// accidental double counting.
    /// @{
    SummaryInvocation(const SummaryInvocation &) = delete;
    SummaryInvocation &operator=(const SummaryInvocation &) = delete;
    /// @}

  private:
    SummaryStatisticsSP m_stats;
    ElapsedTime m_elapsed_time;
  };

private:
  void OnInvoked() noexcept { m_count.fetch_add(1, std::memory_order_relaxed); }
  lldb_private::StatsDuration m_total_time;
  const std::string m_impl_type;
  const std::string m_name;
  std::atomic<uint64_t> m_count;
};

/// A class that wraps a std::map of SummaryStatistics objects behind a mutex.
class SummaryStatisticsCache {
public:
  /// Get the SummaryStatistics object for a given provider name, or insert
  /// if statistics for that provider is not in the map.
  SummaryStatisticsSP
  GetSummaryStatisticsForProvider(lldb_private::TypeSummaryImpl &provider) {
    std::lock_guard<std::mutex> guard(m_map_mutex);
    if (auto iterator = m_summary_stats_map.find(provider.GetName());
        iterator != m_summary_stats_map.end())
      return iterator->second;

    auto it = m_summary_stats_map.try_emplace(
        provider.GetName(),
        std::make_shared<SummaryStatistics>(provider.GetName(),
                                            provider.GetSummaryKindName()));
    return it.first->second;
  }

  llvm::json::Value ToJSON();

  void Reset();

private:
  llvm::StringMap<SummaryStatisticsSP> m_summary_stats_map;
  std::mutex m_map_mutex;
};

/// A class that represents statistics for a since lldb_private::Target.
class TargetStats {
public:
  llvm::json::Value ToJSON(Target &target,
                           const lldb_private::StatisticsOptions &options);

  void SetLaunchOrAttachTime();
  void SetFirstPrivateStopTime();
  void SetFirstPublicStopTime();
  void IncreaseSourceMapDeduceCount();
  void IncreaseSourceRealpathAttemptCount(uint32_t count);
  void IncreaseSourceRealpathCompatibleCount(uint32_t count);

  StatsDuration &GetCreateTime() { return m_create_time; }
  StatsSuccessFail &GetExpressionStats() { return m_expr_eval; }
  StatsSuccessFail &GetFrameVariableStats() { return m_frame_var; }
  void Reset(Target &target);

protected:
  StatsDuration m_create_time;
  std::optional<StatsTimepoint> m_launch_or_attach_time;
  std::optional<StatsTimepoint> m_first_private_stop_time;
  std::optional<StatsTimepoint> m_first_public_stop_time;
  StatsSuccessFail m_expr_eval{"expressionEvaluation"};
  StatsSuccessFail m_frame_var{"frameVariable"};
  std::vector<intptr_t> m_module_identifiers;
  uint32_t m_source_map_deduce_count = 0;
  uint32_t m_source_realpath_attempt_count = 0;
  uint32_t m_source_realpath_compatible_count = 0;
  void CollectStats(Target &target);
};

class DebuggerStats {
public:
  static void SetCollectingStats(bool enable) { g_collecting_stats = enable; }
  static bool GetCollectingStats() { return g_collecting_stats; }

  /// Get metrics associated with one or all targets in a debugger in JSON
  /// format.
  ///
  /// \param debugger
  ///   The debugger to get the target list from if \a target is NULL.
  ///
  /// \param target
  ///   The single target to emit statistics for if non NULL, otherwise dump
  ///   statistics only for the specified target.
  ///
  /// \param summary_only
  ///   If true, only report high level summary statistics without
  ///   targets/modules/breakpoints etc.. details.
  ///
  /// \return
  ///     Returns a JSON value that contains all target metrics.
  static llvm::json::Value
  ReportStatistics(Debugger &debugger, Target *target,
                   const lldb_private::StatisticsOptions &options);

  /// Reset metrics associated with one or all targets in a debugger.
  ///
  /// \param debugger
  ///   The debugger to reset the target list from if \a target is NULL.
  ///
  /// \param target
  ///   The target to reset statistics for, or if null, reset statistics
  ///   for all targets
  static void ResetStatistics(Debugger &debugger, Target *target);

protected:
  // Collecting stats can be set to true to collect stats that are expensive
  // to collect. By default all stats that are cheap to collect are enabled.
  // This settings is here to maintain compatibility with "statistics enable"
  // and "statistics disable".
  static bool g_collecting_stats;
};

} // namespace lldb_private

#endif // LLDB_TARGET_STATISTICS_H
