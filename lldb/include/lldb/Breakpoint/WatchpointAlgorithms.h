//===-- WatchpointAlgorithms.h ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_WATCHPOINTALGORITHMS_H
#define LLDB_BREAKPOINT_WATCHPOINTALGORITHMS_H

#include "lldb/Breakpoint/WatchpointResource.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/lldb-public.h"

#include <vector>

namespace lldb_private {

class WatchpointAlgorithms {

public:
  static std::vector<lldb::WatchpointResourceSP> AtomizeWatchpointRequest(
      lldb::addr_t addr, size_t size, bool read, bool write,
      lldb::WatchpointHardwareFeature supported_features, ArchSpec &arch);

  // Should be protected, but giving access to the algorithms in the unit
  // tests is not easy, so it's public.
  static std::vector<std::pair<lldb::addr_t, size_t>>
  PowerOf2Watchpoints(lldb::addr_t user_addr, size_t user_size,
                      size_t min_byte_size, size_t max_byte_size,
                      uint32_t address_byte_size);
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_WATCHPOINTALGORITHMS_H
