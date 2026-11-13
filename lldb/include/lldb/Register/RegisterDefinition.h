//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Publication of external register definitions as descriptions and layouts.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERDEFINITION_H
#define LLDB_REGISTER_REGISTERDEFINITION_H

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace lldb_private {

/// Mutable input for one register in an external packet or memory image.
///
/// Offsets describe that input format, while relationship references are
/// positions in the same input batch. Publication resolves relationships to
/// logical IDs and separates metadata from bank placement. Type pointers are
/// borrowed; pass their owning nodes to CreateRegisterImageLayout().
struct RegisterDefinition {
  ConstString name;
  ConstString alt_name;
  ConstString set_name;
  uint32_t byte_size = LLDB_INVALID_INDEX32;
  uint32_t byte_offset = LLDB_INVALID_INDEX32;
  lldb::Encoding encoding = lldb::eEncodingUint;
  lldb::Format format = lldb::eFormatHex;
  uint32_t regnum_dwarf = LLDB_INVALID_REGNUM;
  uint32_t regnum_ehframe = LLDB_INVALID_REGNUM;
  uint32_t regnum_generic = LLDB_INVALID_REGNUM;
  uint32_t regnum_remote = LLDB_INVALID_REGNUM;
  // Input references are positions in this one input batch. Finalization
  // resolves them to IDs before publishing the description.
  std::vector<uint32_t> value_registers;
  std::vector<uint32_t> invalidate_registers;
  uint32_t value_reg_offset = 0;
  const RegisterType *type = nullptr;
  RegisterID id = LLDB_INVALID_REGNUM;
};

/// Parse a structured external definition and publish a bound image layout. The
/// resulting description owns names and types. Return nullptr when input
/// parsing or finalization fails.
std::shared_ptr<const RegisterLayout>
CreateRegisterImageLayout(const StructuredData::Dictionary &dict,
                          const ArchSpec &arch);

/// Publish input definitions as an owning description and a bank-zero image.
/// Resolve input relationship indices before exposing the result. Retain
/// \p types for the description lifetime. Return nullptr on invalid input.
std::shared_ptr<const RegisterLayout>
CreateRegisterImageLayout(std::vector<RegisterDefinition> registers,
                          const ArchSpec &arch,
                          RegisterDescription::OwnedTypes types = {});

/// Copy the descriptors, apply the update, then rebuild and validate image
/// placement. Preserve groups and owned types; leave the original layout
/// unchanged. The input layout must be finalized. Invalid updated metadata or
/// placement is returned as an error.
llvm::Expected<RegisterLayout> ReconfigureRegisterImageLayout(
    const RegisterLayout &layout,
    llvm::function_ref<void(llvm::MutableArrayRef<RegisterDescriptor>)> update);

/// Return an offset only for a complete contiguous value in image bank zero.
/// The input layout must be finalized.
/// Return LLDB_INVALID_INDEX32 for missing, split, constant, or unavailable
/// values; a logical ID is never treated as an offset.
uint32_t GetRegisterImageOffset(const RegisterLayout &layout, RegisterID id);

/// Append a derived register and connect its cache invalidation relationships.
/// The value_registers list must be nonempty and contain valid positions in the
/// input batch; this helper runs before finalization converts them to IDs.
void addSupplementaryRegister(std::vector<RegisterDefinition> &registers,
                              RegisterDefinition reg);

} // namespace lldb_private
#endif
