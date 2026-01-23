//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_VIRTUALDATAEXTRACTOR_H
#define LLDB_UTILITY_VIRTUALDATAEXTRACTOR_H

#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/lldb-types.h"

namespace lldb_private {

/// A DataExtractor subclass that allows reading data at virtual addresses
/// using a lookup table that maps virtual address ranges to physical offsets.
///
/// This class maintains a lookup table where each entry contains:
/// - base: starting virtual address for this entry
/// - size: size of this entry in bytes
/// - data: physical offset in the underlying data buffer
///
/// Reads are translated from virtual addresses to physical offsets using
/// this lookup table. Reads cannot cross entry boundaries and this is
/// enforced with assertions.
class VirtualDataExtractor : public DataExtractor {
public:
  /// Type alias for the range map used internally.
  /// Maps virtual addresses (base) to physical offsets (data).
  using LookupTable =
      RangeDataVector<lldb::offset_t, lldb::offset_t, lldb::offset_t>;

  VirtualDataExtractor() = default;

  VirtualDataExtractor(const void *data, lldb::offset_t data_length,
                       lldb::ByteOrder byte_order, uint32_t addr_size,
                       LookupTable lookup_table);

  VirtualDataExtractor(const lldb::DataBufferSP &data_sp,
                       lldb::ByteOrder byte_order, uint32_t addr_size,
                       LookupTable lookup_table);

  VirtualDataExtractor(const lldb::DataBufferSP &data_sp,
                       LookupTable lookup_table);

  const void *GetData(lldb::offset_t *offset_ptr,
                      lldb::offset_t length) const override;

  const uint8_t *PeekData(lldb::offset_t offset,
                          lldb::offset_t length) const override;

  lldb::DataExtractorSP GetSubsetExtractorSP(lldb::offset_t offset,
                                             lldb::offset_t length) override;

  lldb::DataExtractorSP GetSubsetExtractorSP(lldb::offset_t offset) override;

  llvm::ArrayRef<uint8_t> GetData() const override;

  /// Unchecked overrides
  /// @{
  uint8_t GetU8_unchecked(lldb::offset_t *offset_ptr) const override;
  uint16_t GetU16_unchecked(lldb::offset_t *offset_ptr) const override;
  uint32_t GetU32_unchecked(lldb::offset_t *offset_ptr) const override;
  uint64_t GetU64_unchecked(lldb::offset_t *offset_ptr) const override;
  /// @}

protected:
  /// Find the lookup entry that contains the given virtual address.
  const LookupTable::Entry *FindEntry(lldb::offset_t virtual_addr) const;

  /// Validate that a read at a virtual address is within bounds and
  /// does not cross entry boundaries.
  bool ValidateVirtualRead(lldb::offset_t virtual_addr,
                           lldb::offset_t length) const;

private:
  LookupTable m_lookup_table;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_VIRTUALDATAEXTRACTOR_H
