//===-- WatchpointResource.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_WATCHPOINTRESOURCE_H
#define LLDB_TARGET_WATCHPOINTRESOURCE_H

#include "lldb/lldb-public.h"

#include <set>

namespace lldb_private {

class WatchpointResource
    : public std::enable_shared_from_this<WatchpointResource> {

public:
  // Constructors and Destructors
  WatchpointResource(lldb::addr_t addr, size_t size, bool read, bool write);

  ~WatchpointResource();

  void GetResourceMemoryRange(lldb::addr_t &addr, size_t &size) const;

  void GetResourceType(bool &read, bool &write) const;

  void RegisterWatchpoint(lldb::WatchpointSP &wp_sp);

  void DeregisterWatchpoint(lldb::WatchpointSP &wp_sp);

  size_t GetNumDependantWatchpoints();

  bool DependantWatchpointsContains(lldb::WatchpointSP wp_sp_to_match);

private:
  WatchpointResource(const WatchpointResource &) = delete;
  const WatchpointResource &operator=(const WatchpointResource &) = delete;

  // start address & size aligned & expanded to be a valid watchpoint
  // memory granule on this target.
  lldb::addr_t m_addr;
  size_t m_size;

  bool m_watch_read;  // true if we stop when the watched data is read from
  bool m_watch_write; // true if we stop when the watched data is written to

  // The watchpoints using this WatchpointResource.
  std::set<lldb::WatchpointSP> m_watchpoints;
};

} // namespace lldb_private

#endif // LLDB_TARGET_WATCHPOINTRESOURCE_H
