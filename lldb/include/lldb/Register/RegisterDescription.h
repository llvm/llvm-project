//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Owning register metadata, logical identities, and external numbering maps.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERDESCRIPTION_H
#define LLDB_REGISTER_REGISTERDESCRIPTION_H

#include "lldb/Register/RegisterType.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lldb_private {

/// A logical register identity scoped to one description, independent of an
/// exposed table index, external register number, or storage offset.
using RegisterID = uint32_t;

/// Metadata for one logical register, independent of where its bytes are saved.
///
/// Relationships contain RegisterIDs from the same description, not array
/// indices. A single value_registers parent describes an alias slice; multiple
/// parents describe an ordered composite. Type pointers are borrowed from type
/// nodes retained by the owning description.
struct RegisterDescriptor {
  using Numbering = std::array<uint32_t, lldb::kNumRegisterKinds>;
  RegisterID id = LLDB_INVALID_REGNUM;
  std::string name;
  std::string alt_name;
  uint32_t byte_size = 0;
  lldb::Encoding encoding = lldb::eEncodingUint;
  lldb::Format format = lldb::eFormatHex;

  /// External numbers indexed by RegisterKind; absent mappings use
  /// LLDB_INVALID_REGNUM. These numbers are distinct from the logical ID.
  Numbering numbers;
  std::vector<RegisterID> value_registers;

  /// Other cached values that become stale when this register is written.
  std::vector<RegisterID> invalidate_registers;

  /// Least-significant byte position within a single parent logical value.
  uint32_t value_byte_offset = 0;
  const RegisterType *type = nullptr;

  /// Writing this value clears the remaining high bytes of its parent.
  RegisterID zero_extend_parent = LLDB_INVALID_REGNUM;

  /// Little-endian bytes of an architectural constant. Empty for ordinary
  /// registers; zero registers need no saved-state bank.
  std::vector<uint8_t> constant_value;

  RegisterDescriptor() { numbers.fill(LLDB_INVALID_REGNUM); }

  RegisterDescriptor(RegisterID id, llvm::StringRef name, const char *alt_name,
                     uint32_t byte_size, lldb::Encoding encoding,
                     lldb::Format format, Numbering numbers,
                     llvm::ArrayRef<RegisterID> value_registers = {},
                     llvm::ArrayRef<RegisterID> invalidate_registers = {},
                     const RegisterType *type = nullptr)
      : id(id), name(name.str()), alt_name(alt_name ? alt_name : ""),
        byte_size(byte_size), encoding(encoding), format(format),
        numbers(numbers),
        value_registers(value_registers.begin(), value_registers.end()),
        invalidate_registers(invalidate_registers.begin(),
                             invalidate_registers.end()),
        type(type) {}
};

/// A named presentation group of logical register IDs. Group membership and
/// order are independent of physical bank placement.
struct RegisterGroup {
  std::string name;
  std::string short_name;
  std::vector<RegisterID> registers;
};

/// Finalized, owning metadata shared by layouts and register contexts.
///
/// An architecture describes names, logical identities, numbering, types, and
/// value relationships here. OS and file-format offsets belong in
/// RegisterLayout; process access belongs in a RegisterContext implementation.
/// Borrowed views and descriptor pointers remain valid while this description
/// is unchanged.
///
/// Create() validates the input and builds lookup maps before publication. Keep
/// the finalized object shared and const when exposing it to consumers.
class RegisterDescription {
public:
  /// Own every type node referenced by a description, including enum fields.
  using OwnedTypes = std::vector<std::shared_ptr<const RegisterType>>;

  /// Validate and take ownership of registers, groups, and referenced type
  /// nodes. Reject invalid sizes, duplicate IDs, missing relationships, cycles,
  /// and invalid group members. When names or external numbers are shared by
  /// aliases, lookup preserves the first matching register in input order.
  static llvm::Expected<RegisterDescription>
  Create(std::vector<RegisterDescriptor> registers,
         std::vector<RegisterGroup> groups = {}, OwnedTypes types = {});

  llvm::ArrayRef<RegisterDescriptor> GetRegisters() const {
    return m_registers;
  }
  llvm::ArrayRef<RegisterGroup> GetGroups() const { return m_groups; }
  llvm::ArrayRef<std::shared_ptr<const RegisterType>> GetOwnedTypes() const {
    return m_owned_types;
  }
  llvm::ArrayRef<RegisterID> GetUserRegisters() const {
    return m_user_registers;
  }

  /// Copy the metadata with a new user-visible order or subset of registers.
  /// IDs and storage order are preserved. Unknown or repeated IDs are errors.
  llvm::Expected<RegisterDescription>
  WithUserRegisters(llvm::ArrayRef<RegisterID> order) const;

  /// Return a descriptor by its stored table index, or nullptr when out of
  /// range.
  const RegisterDescriptor *GetRegisterAtIndex(size_t index) const {
    return index < m_registers.size() ? &m_registers[index] : nullptr;
  }

  /// Return a presentation group, or nullptr when the index is out of range.
  const RegisterGroup *GetGroupAtIndex(size_t index) const {
    return index < m_groups.size() ? &m_groups[index] : nullptr;
  }

  /// Look up a logical ID; return nullptr if it is not in this description.
  const RegisterDescriptor *Find(RegisterID id) const;

  /// Find an exact, case-sensitive name, then an alternate name. Return nullptr
  /// if neither lookup succeeds.
  const RegisterDescriptor *Find(llvm::StringRef name) const;

  /// Resolve external numbering to a logical value, never a table index.
  const RegisterDescriptor *FindByNumber(lldb::RegisterKind kind,
                                         uint32_t number) const;

  /// Debugger name lookup: generic roles take precedence over aliases, then
  /// names and alternate names are compared without regard to case.
  const RegisterDescriptor *FindByName(llvm::StringRef name,
                                       size_t start_index = 0) const;

  /// Explicit presentation index for clients enumerating this description.
  std::optional<size_t> GetRegisterIndex(RegisterID id) const;

  /// Convert through a logical register identity. Return LLDB_INVALID_REGNUM
  /// when either numbering scheme or the requested mapping is unavailable.
  uint32_t ConvertNumber(lldb::RegisterKind source_kind, uint32_t number,
                         lldb::RegisterKind target_kind) const;

  /// Find a group by its exact short name, or return nullptr.
  const RegisterGroup *FindGroup(llvm::StringRef short_name) const;

  /// Return the first presentation group containing the ID, or nullptr.
  const RegisterGroup *FindGroupForRegister(RegisterID id) const;
  bool IsInGroup(RegisterID id, llvm::StringRef short_name) const;

private:
  OwnedTypes m_owned_types;
  std::vector<RegisterDescriptor> m_registers;
  std::vector<RegisterGroup> m_groups;
  std::vector<RegisterID> m_user_registers;
  std::map<RegisterID, size_t> m_ids;
  std::map<std::string, size_t, std::less<>> m_names;
  std::array<std::map<uint32_t, size_t>, lldb::kNumRegisterKinds> m_numbers;
};

} // namespace lldb_private

#endif
