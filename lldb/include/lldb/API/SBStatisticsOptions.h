//===-- SBStatisticsOptions.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBSTATISTICSOPTIONS_H
#define LLDB_API_SBSTATISTICSOPTIONS_H

#include "lldb/API/SBDefines.h"

namespace lldb {

/// This class handles the verbosity when dumping statistics
class LLDB_API SBStatisticsOptions {
public:
  SBStatisticsOptions();
  SBStatisticsOptions(const lldb::SBStatisticsOptions &rhs);
  ~SBStatisticsOptions();

  const SBStatisticsOptions &operator=(const lldb::SBStatisticsOptions &rhs);

  void SetSummaryOnly(bool b);
  bool GetSummaryOnly();

  /// If set to true, the debugger will load all debug info that is available
  /// and report statistics on the total amount. If this is set to false, then
  /// only report statistics on the currently loaded debug information.
  /// This can avoid loading debug info from separate files just so it can
  /// report the total size which can slow down statistics reporting.
  void SetReportAllAvailableDebugInfo(bool b);
  bool GetReportAllAvailableDebugInfo();

protected:
  friend class SBTarget;
  const lldb_private::StatisticsOptions &ref() const;

private:
  std::unique_ptr<lldb_private::StatisticsOptions> m_opaque_up;
};
} // namespace lldb
#endif // LLDB_API_SBSTATISTICSOPTIONS_H
