//===-- MemoryRegionInfoCache.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_MEMORYREGIONINFOCACHE_H
#define LLDB_TARGET_MEMORYREGIONINFOCACHE_H

#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/RangeMap.h"

namespace lldb_private {
class MemoryRegionInfoCache {
public:
  MemoryRegionInfoCache(Process &process)
      : m_region_infos(), m_process(process) {}

  /// Remove all cached entries.
  void Clear();

  /// Remove cached information about region containing \a addr, if any.
  void Flush(lldb::addr_t addr, lldb::addr_t size);

  /// Locate the memory region that contains load_addr.
  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &region_info);

  void AddRegion(const MemoryRegionInfo &region_info);

private:
  typedef RangeDataVector<lldb::addr_t, lldb::addr_t,
                          lldb_private::MemoryRegionInfo>
      InfoMap;
  InfoMap m_region_infos;
  Process &m_process;
};
} // namespace lldb_private

#endif // LLDB_TARGET_MEMORYREGIONINFOCACHE_H
