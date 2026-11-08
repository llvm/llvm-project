//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERDEFINITION_H
#define LLDB_REGISTER_REGISTERDEFINITION_H

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace lldb_private {

/// Temporary external definition. Its offsets are facts about a packet or
/// memory image. Input relationship indices are resolved to IDs at publication.
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

std::shared_ptr<const RegisterLayout>
CreateRegisterImageLayout(const StructuredData::Dictionary &dict,
                          const ArchSpec &arch);
std::shared_ptr<const RegisterLayout>
CreateRegisterImageLayout(std::vector<RegisterDefinition> registers,
                          const ArchSpec &arch,
                          RegisterDescription::OwnedTypes types = {});
llvm::Expected<RegisterLayout> ReconfigureRegisterImageLayout(
    const RegisterLayout &layout,
    llvm::function_ref<void(llvm::MutableArrayRef<RegisterDescriptor>)> update);
uint32_t GetRegisterImageOffset(const RegisterLayout &layout, RegisterID id);
void addSupplementaryRegister(std::vector<RegisterDefinition> &registers,
                              RegisterDefinition reg);

} // namespace lldb_private
#endif
