//===-- CoreFileMemoryRanges.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RangeMap.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/AddressRanges.h"

#ifndef LLDB_TARGET_COREFILEMEMORYRANGES_H
#define LLDB_TARGET_COREFILEMEMORYRANGES_H

namespace lldb_private {

struct CoreFileMemoryRange {
  llvm::AddressRange range;  /// The address range to save into the core file.
  uint32_t lldb_permissions; /// A bit set of lldb::Permissions bits.

  bool operator==(const CoreFileMemoryRange &rhs) const {
    return range == rhs.range && lldb_permissions == rhs.lldb_permissions;
  }

  bool operator!=(const CoreFileMemoryRange &rhs) const {
    return !(*this == rhs);
  }

  bool operator<(const CoreFileMemoryRange &rhs) const {
    return std::tie(range, lldb_permissions) <
           std::tie(rhs.range, rhs.lldb_permissions);
  }

  std::string Dump() const {
    lldb_private::StreamString stream;
    stream << "[";
    stream.PutHex64(range.start());
    stream << '-';
    stream.PutHex64(range.end());
    stream << ")";
    return stream.GetString().str();
  }
};

class CoreFileMemoryRanges
    : public lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t,
                                           CoreFileMemoryRange> {
public:
  /// Finalize and merge all overlapping ranges in this collection. Ranges
  /// will be separated based on permissions.
  Status FinalizeCoreFileSaveRanges();
};
} // namespace lldb_private

#endif // LLDB_TARGET_COREFILEMEMORYRANGES_H
