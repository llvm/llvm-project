//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERDESCRIPTIONBUILDER_H
#define LLDB_REGISTER_REGISTERDESCRIPTIONBUILDER_H

#include "lldb/Register/RegisterDescription.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <set>
#include <utility>

namespace lldb_private {

/// Define a value with explicit EH-frame, DWARF, and generic numbers. The
/// logical ID supplies the initial LLDB number; configurations may project
/// their exposed numbering separately. Process-plugin numbers are adapter data.
inline RegisterDescriptor
DefineRegister(RegisterID id, llvm::StringRef name, const char *alt_name,
               uint32_t byte_size, uint32_t ehframe, uint32_t dwarf,
               uint32_t generic = LLDB_INVALID_REGNUM,
               lldb::Encoding encoding = lldb::eEncodingUint,
               lldb::Format format = lldb::eFormatHex,
               llvm::ArrayRef<RegisterID> parents = {},
               llvm::ArrayRef<RegisterID> invalidates = {}) {
  return RegisterDescriptor(id, name, alt_name, byte_size, encoding, format,
                            {ehframe, dwarf, generic, LLDB_INVALID_REGNUM, id},
                            parents, invalidates);
}

inline RegisterDescriptor WithAliasProperties(RegisterDescriptor reg,
                                              uint32_t byte_offset,
                                              bool zero_extends_parent) {
  assert(!zero_extends_parent || reg.value_registers.size() == 1);

  reg.value_byte_offset = byte_offset;

  if (zero_extends_parent)
    reg.zero_extend_parent = reg.value_registers.front();

  return reg;
}

inline RegisterDescriptor WithConstantZero(RegisterDescriptor reg) {
  reg.constant_value.resize(reg.byte_size, 0);

  return reg;
}

inline RegisterGroup MakeRegisterGroup(const char *name, const char *short_name,
                                       llvm::ArrayRef<RegisterID> members) {
  return {name, short_name,
          std::vector<RegisterID>(members.begin(), members.end())};
}

/// Select values in catalog order, independently of their presentation groups.
/// Relationships to values excluded by the configuration are not published.
inline std::vector<RegisterDescriptor>
SelectRegisterDefinitions(llvm::ArrayRef<RegisterDescriptor> definitions,
                          llvm::ArrayRef<RegisterGroup> groups) {
  std::set<RegisterID> selected;

  for (const auto &group : groups)
    selected.insert(group.registers.begin(), group.registers.end());

  std::vector<RegisterDescriptor> values;

  for (auto reg : definitions) {
    if (!selected.count(reg.id))
      continue;

    for (auto *references : {&reg.value_registers, &reg.invalidate_registers}) {
      assert(llvm::all_of(*references, [&](RegisterID id) {
        return llvm::any_of(definitions, [&](const auto &candidate) {
          return candidate.id == id;
        });
      }));

      llvm::erase_if(*references,
                     [&](RegisterID id) { return !selected.count(id); });
    }

    if (!selected.count(reg.zero_extend_parent))
      reg.zero_extend_parent = LLDB_INVALID_REGNUM;

    values.push_back(std::move(reg));
  }

  return values;
}

/// Project exposed widths without choosing any saved-register offsets.
inline void
SetRegisterByteSizes(std::vector<RegisterDescriptor> &registers,
                     llvm::ArrayRef<std::pair<RegisterID, uint32_t>> sizes) {
  for (auto &reg : registers)
    for (const auto &[id, size] : sizes)
      if (reg.id == id)
        reg.byte_size = std::min(reg.byte_size, size);
}

inline RegisterDescription
HideUserRegisters(const RegisterDescription &description,
                  llvm::ArrayRef<RegisterID> hidden) {
  std::vector<RegisterID> visible;

  for (RegisterID id : description.GetUserRegisters())
    if (!llvm::is_contained(hidden, id))
      visible.push_back(id);

  return llvm::cantFail(description.WithUserRegisters(visible));
}

} // namespace lldb_private

#endif
