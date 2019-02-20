//===-- NativeWatchpointList.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeWatchpointList_h_
#define liblldb_NativeWatchpointList_h_

#include "lldb/Utility/Status.h"
#include "lldb/lldb-private-forward.h"

#include <map>

namespace lldb_private {
struct NativeWatchpoint {
  lldb::addr_t m_addr;
  size_t m_size;
  uint32_t m_watch_flags;
  bool m_hardware;
};

class NativeWatchpointList {
public:
  Status Add(lldb::addr_t addr, size_t size, uint32_t watch_flags,
             bool hardware);

  Status Remove(lldb::addr_t addr);

  using WatchpointMap = std::map<lldb::addr_t, NativeWatchpoint>;

  const WatchpointMap &GetWatchpointMap() const;

private:
  WatchpointMap m_watchpoints;
};
}

#endif // ifndef liblldb_NativeWatchpointList_h_
