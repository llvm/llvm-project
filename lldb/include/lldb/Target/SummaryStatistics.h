//===-- SummaryStatistics.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_SUMMARYSTATISTICS_H
#define LLDB_TARGET_SUMMARYSTATISTICS_H


#include "lldb/Target/Statistics.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

class SummaryStatistics {
public:
  SummaryStatistics(lldb_private::ConstString name) : 
    m_total_time(), m_name(name), m_summary_count(0) {}

  lldb_private::StatsDuration &GetDurationReference();

  lldb_private::ConstString GetName() const;

  uint64_t GetSummaryCount() const;

private:
   lldb_private::StatsDuration m_total_time;
   lldb_private::ConstString m_name;
   uint64_t m_summary_count;
};

} // namespace lldb_private

#endif // LLDB_TARGET_SUMMARYSTATISTICS_H
