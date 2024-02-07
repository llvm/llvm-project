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

protected:
  friend class SBTarget;
  const lldb_private::StatisticsOptions &ref() const;

private:
  std::unique_ptr<lldb_private::StatisticsOptions> m_opaque_up;
};
} // namespace lldb
#endif // LLDB_API_SBSTATISTICSOPTIONS_H
