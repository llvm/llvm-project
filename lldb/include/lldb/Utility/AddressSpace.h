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

/// A single address space reported by a process.
struct AddressSpaceInfo {
  std::string name;
  lldb::addr_space_t space_id = 0;
  bool is_thread_specific = false;

  /// Optional flat aperture: pointers whose pointee has DWARF address class
  /// \a dwarf_address_class hold a value in units of \a pointer_unit bytes
  /// that maps to the process address flat_base + value * pointer_unit. This
  /// lets value objects dereference pointers into memories that are not byte
  /// addressed or not in the default address space (e.g. GPU, DSP, or
  /// accelerator). Address class 0 (default / flat) has no aperture; 0 is the
  /// NULL pointer.
  std::optional<uint32_t> dwarf_address_class;
  std::optional<lldb::addr_t> flat_base;
  /// Size in bytes of the flat aperture; 0 means unbounded.
  uint64_t flat_size = 0;
  uint64_t pointer_unit = 1;
};

bool fromJSON(const llvm::json::Value &value, AddressSpaceInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AddressSpaceInfo &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ADDRESSSPACE_H
