//===-- AddressSpace.h ----------------------------------------------------===//
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
#include <string>
#include <vector>

namespace lldb_private {

/// Describes a single address space reported by a process. Processes such as
/// GPUs can have multiple address spaces (for example global, local, private or
/// generic memory) where the same numeric address refers to different storage
/// depending on the address space. See the "jAddressSpacesInfo" packet in
/// docs/resources/lldbgdbremote.md for the wire format.
struct AddressSpaceInfo {
  std::string name;        ///< The name of the address space.
  uint64_t value;          ///< The integer identifier of the address space.
  bool is_thread_specific; ///< True if the address space is thread specific.
};

bool fromJSON(const llvm::json::Value &value, AddressSpaceInfo &data,
              llvm::json::Path path);

llvm::json::Value toJSON(const AddressSpaceInfo &data);

} // namespace lldb_private

#endif // LLDB_UTILITY_ADDRESSSPACE_H
