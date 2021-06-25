//===-- MemoryTagManager.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_MEMORYTAGMANAGER_H
#define LLDB_TARGET_MEMORYTAGMANAGER_H

#include "lldb/Utility/RangeMap.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Error.h"

namespace lldb_private {

// This interface allows high level commands to handle memory tags
// in a generic way.
//
// Definitions:
//   logical tag    - the tag stored in a pointer
//   allocation tag - the tag stored in hardware
//                    (e.g. special memory, cache line bits)
//   granule        - number of bytes of memory a single tag applies to

class MemoryTagManager {
public:
  typedef Range<lldb::addr_t, lldb::addr_t> TagRange;

  // Extract the logical tag from a pointer
  // The tag is returned as a plain value, with any shifts removed.
  // For example if your tags are stored in bits 56-60 then the logical tag
  // you get will have been shifted down 56 before being returned.
  virtual lldb::addr_t GetLogicalTag(lldb::addr_t addr) const = 0;

  // Remove non address bits from a pointer
  virtual lldb::addr_t RemoveNonAddressBits(lldb::addr_t addr) const = 0;

  // Return the difference between two addresses, ignoring any logical tags they
  // have. If your tags are just part of a larger set of ignored bits, this
  // should ignore all those bits.
  virtual ptrdiff_t AddressDiff(lldb::addr_t addr1,
                                lldb::addr_t addr2) const = 0;

  // Return the number of bytes a single tag covers
  virtual lldb::addr_t GetGranuleSize() const = 0;

  // Align an address range to granule boundaries.
  // So that reading memory tags for the new range returns
  // tags that will cover the original range.
  //
  // Say your granules are 16 bytes and you want
  // tags for 16 bytes of memory starting from address 8.
  // 1 granule isn't enough because it only covers addresses
  // 0-16, we want addresses 8-24. So the range must be
  // expanded to 2 granules.
  virtual TagRange ExpandToGranule(TagRange range) const = 0;

  // Return the type value to use in GDB protocol qMemTags packets to read
  // allocation tags. This is named "Allocation" specifically because the spec
  // allows for logical tags to be read the same way, though we do not use that.
  //
  // This value is unique within a given architecture. Meaning that different
  // tagging schemes within the same architecture should use unique values,
  // but other architectures can overlap those values.
  virtual int32_t GetAllocationTagType() const = 0;

  // Return the number of bytes a single tag will be packed into during
  // transport. For example an MTE tag is 4 bits but occupies 1 byte during
  // transport.
  virtual size_t GetTagSizeInBytes() const = 0;

  // Unpack tags from their stored format (e.g. gdb qMemTags data) into seperate
  // tags. Checks that each tag is within the expected value range and that the
  // number of tags found matches the number of granules we originally asked
  // for.
  virtual llvm::Expected<std::vector<lldb::addr_t>>
  UnpackTagsData(const std::vector<uint8_t> &tags, size_t granules) const = 0;

  virtual ~MemoryTagManager() {}
};

} // namespace lldb_private

#endif // LLDB_TARGET_MEMORYTAGMANAGER_H
