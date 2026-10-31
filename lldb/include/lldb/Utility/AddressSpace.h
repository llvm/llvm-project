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

/// Optional flat aperture: pointers whose pointee lives in this address space
/// hold a value in units of \a pointer_unit bytes that maps to the default
/// process address space at base + value * pointer_unit. This lets value
/// objects dereference pointers into memories that are not byte-addressed or
/// are mapped into a flat aperture of the process address space (e.g. GPU,
/// DSP, or accelerator).
struct FlatAperture {
  lldb::addr_t base = 0;
  /// Size in bytes of the flat aperture; 0 means unbounded.
  uint64_t size = 0;
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
  std::optional<FlatAperture> flat_aperture;
};

bool fromJSON(const llvm::json::Value &value, AddressSpaceInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AddressSpaceInfo &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ADDRESSSPACE_H
