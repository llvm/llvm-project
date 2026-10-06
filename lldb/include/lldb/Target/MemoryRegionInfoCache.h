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

#include <map>
#include <mutex>
#include <optional>

namespace lldb_private {
class MemoryRegionInfoCache {
public:
  MemoryRegionInfoCache() = default;

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
  std::map<lldb::addr_t, MemoryRegionInfo> m_region_infos;
  std::mutex m_mutex;
};
} // namespace lldb_private

#endif // LLDB_TARGET_MEMORYREGIONINFOCACHE_H
