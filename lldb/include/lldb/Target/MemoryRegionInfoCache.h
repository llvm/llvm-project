//===----------------------------------------------------------------------===//
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

#include <mutex>
#include <optional>

namespace lldb_private {
class MemoryRegionInfoCache {
public:
  MemoryRegionInfoCache() : m_region_infos(), m_is_sorted(true), m_mutex() {}

  /// Remove all cached entries.  Should be called whenever
  /// Process resumes execution of the inferior.
  void Clear();

  /// Return a MemoryRegionInfo that covers \p load_addr,
  /// returns empty optional if there is no entry.
  std::optional<MemoryRegionInfo> GetMemoryRegion(lldb::addr_t load_addr);

  /// Add a MemoryRegionInfo to the collection.
  void AddRegion(const MemoryRegionInfo &region_info);

  size_t GetSize();

private:
  typedef RangeDataVector<lldb::addr_t, size_t, lldb_private::MemoryRegionInfo>
      InfoMap;
  InfoMap m_region_infos;
  bool m_is_sorted;
  std::mutex m_mutex;
};
} // namespace lldb_private

#endif // LLDB_TARGET_MEMORYREGIONINFOCACHE_H
