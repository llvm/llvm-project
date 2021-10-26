//===-- Statistics.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_STATISTICS_H
#define LLDB_TARGET_STATISTICS_H

#include <chrono>
#include <string>
#include <vector>

#include "lldb/Utility/Stream.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/JSON.h"

namespace lldb_private {

using StatsClock = std::chrono::high_resolution_clock;
using StatsDuration = std::chrono::duration<double>;
using StatsTimepoint = std::chrono::time_point<StatsClock>;

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
    StatsDuration elapsed = StatsClock::now() - m_start_time;
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
  double symtab_parse_time = 0.0;
  double symtab_index_time = 0.0;
  double debug_parse_time = 0.0;
  double debug_index_time = 0.0;
  uint64_t debug_info_size = 0;
};

/// A class that represents statistics for a since lldb_private::Target.
class TargetStats {
public:
  llvm::json::Value ToJSON(Target &target);

  void SetLaunchOrAttachTime();
  void SetFirstPrivateStopTime();
  void SetFirstPublicStopTime();

  StatsDuration &GetCreateTime() { return m_create_time; }
  StatsSuccessFail &GetExpressionStats() { return m_expr_eval; }
  StatsSuccessFail &GetFrameVariableStats() { return m_frame_var; }

protected:
  StatsDuration m_create_time{0.0};
  llvm::Optional<StatsTimepoint> m_launch_or_attach_time;
  llvm::Optional<StatsTimepoint> m_first_private_stop_time;
  llvm::Optional<StatsTimepoint> m_first_public_stop_time;
  StatsSuccessFail m_expr_eval{"expressionEvaluation"};
  StatsSuccessFail m_frame_var{"frameVariable"};
  std::vector<intptr_t> m_module_identifiers;
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
  /// \return
  ///     Returns a JSON value that contains all target metrics.
  static llvm::json::Value ReportStatistics(Debugger &debugger, Target *target);

protected:
  // Collecting stats can be set to true to collect stats that are expensive
  // to collect. By default all stats that are cheap to collect are enabled.
  // This settings is here to maintain compatibility with "statistics enable"
  // and "statistics disable".
  static bool g_collecting_stats;
};

} // namespace lldb_private

#endif // LLDB_TARGET_STATISTICS_H
