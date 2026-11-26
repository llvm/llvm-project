//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_ADDRESSSPACE_H
#define LLDB_UTILITY_ADDRESSSPACE_H

#include "lldb/lldb-types.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <vector>

namespace lldb_private {

/// Linear mapping from a target address space into the default process address
/// space: a target pointer maps to base + pointer * pointer_unit.
struct FlatAperture {
  lldb::addr_t base = 0;
  /// Size in bytes of the flat aperture; 0 means unbounded.
  uint64_t size = 0;
  /// Byte stride per target pointer unit.
  uint64_t pointer_unit = 1;

  bool operator==(const FlatAperture &rhs) const {
    return base == rhs.base && size == rhs.size &&
           pointer_unit == rhs.pointer_unit;
  }
};

bool fromJSON(const llvm::json::Value &value, FlatAperture &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const FlatAperture &data);

/// A single address space reported by a process.
struct AddressSpaceInfo {
  std::string name;
  lldb::addr_space_t space_id = 0;
  bool is_thread_specific = false;
  std::optional<FlatAperture> flat_aperture = std::nullopt;
};

bool fromJSON(const llvm::json::Value &value, AddressSpaceInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AddressSpaceInfo &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ADDRESSSPACE_H
