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

#include <cinttypes>
#include <cmath>

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

  /// Returns a human readable description of the range suitable for progress
  /// reporting, e.g. "of 1.50MB at 0x00007ffff7d8a000".
  std::string Dump() const {
    lldb_private::StreamString stream;
    stream << "of ";
    const double size = static_cast<double>(range.size());
    constexpr double k = 1000.0;
    constexpr double m = k * k;
    constexpr double g = k * k * k;

    auto formatSize = [&stream](double value, const char *unit) {
      // Omit the decimals if the value rounds to a whole number.
      const uint64_t hundredths = std::llround(value * 100);
      if (hundredths % 100 == 0)
        stream.Printf("%" PRIu64 "%s", hundredths / 100, unit);
      else
        stream.Printf("%.2f%s", value, unit);
    };

    if (size >= g)
      formatSize(size / g, "GB");
    else if (size >= m)
      formatSize(size / m, "MB");
    else if (size >= k)
      formatSize(size / k, "KB");
    else
      stream.Printf("%" PRIu64 "B", range.size());
    stream << " at 0x";
    stream.PutHex64(range.start());
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
