//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Mappings between logical register values and concrete saved-register banks.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERLAYOUT_H
#define LLDB_REGISTER_REGISTERLAYOUT_H

#include "lldb/Register/RegisterDescription.h"
#include "lldb/Register/RegisterValue.h"

namespace lldb_private {

/// A bank identity scoped to one layout, not a register number or byte offset.
using RegisterBankID = uint32_t;

/// A named field in a concrete storage format, independent of table order.
struct RegisterStorageField {
  const char *name;
  uint32_t byte_size;
  uint32_t byte_offset;
};

/// The declared byte extent of one independently acquired storage bank.
struct RegisterBank {
  RegisterBankID id;
  uint64_t byte_size;
};

/// A contiguous range of bank bytes contributing to an encoded register value.
/// value_offset indexes the encoded value in layout byte order, rather than a
/// least-significant-byte offset.
struct RegisterPiece {
  RegisterBankID bank;
  uint64_t byte_offset;
  uint32_t byte_size;
  uint32_t value_offset = 0;
};

/// Storage for one logical value: bank pieces, encoded constant bytes, or an
/// explicitly unavailable location. Piece order does not imply bank order.
struct RegisterLocation {
  enum Kind { Stored, Constant, Unavailable } kind = Unavailable;
  std::vector<RegisterPiece> pieces;
  std::vector<uint8_t> constant;

  /// Unmapped value bytes are zero (for example inactive vector lanes).
  bool zero_fill = false;
};

/// A borrowed read-only bank image. The caller owns the supplied bytes.
struct RegisterData {
  RegisterBankID bank;
  llvm::ArrayRef<uint8_t> bytes;
};

/// A borrowed writable bank image. Writes modify the caller-owned bytes.
struct MutableRegisterData {
  RegisterBankID bank;
  llvm::MutableArrayRef<uint8_t> bytes;
};

/// A concrete saved-register format bound to an immutable description.
///
/// A layout maps logical IDs to banks and pieces; it neither acquires native
/// state nor owns the bank buffers supplied to Read() and Write(). To support a
/// new OS or file format, declare its bank extents and register locations, then
/// finalize the draft with Create() before publishing it to a context.
class RegisterLayout {
public:
  explicit RegisterLayout(lldb::ByteOrder byte_order)
      : m_byte_order(byte_order) {}

  /// Declare a bank in a draft, rejecting duplicate bank IDs. A successful
  /// modification clears any previous description binding.
  llvm::Error AddBank(RegisterBankID bank, uint64_t byte_size);

  /// Add a draft location, rejecting invalid or duplicate logical IDs.
  /// Validation against the description is deferred to Create().
  llvm::Error Add(RegisterID id, RegisterLocation location);

  /// Return the stored location for an ID, or nullptr if it is unmapped.
  const RegisterLocation *Find(RegisterID id) const;

  /// Return a bank declaration, or nullptr if the bank is unknown.
  const RegisterBank *FindBank(RegisterBankID id) const;
  size_t GetBankSize(RegisterBankID id) const {
    const auto *bank = FindBank(id);
    return bank ? bank->byte_size : 0;
  }
  lldb::ByteOrder GetByteOrder() const { return m_byte_order; }

  /// Bind and validate the draft against its owning description. Reject missing
  /// metadata, unknown IDs or banks, out-of-range pieces, and inconsistent
  /// value coverage. A successful result retains the description.
  static llvm::Expected<RegisterLayout>
  Create(std::shared_ptr<const RegisterDescription> description,
         RegisterLayout draft);
  std::shared_ptr<const RegisterDescription> GetDescription() const {
    return m_description;
  }

  /// Whether opaque bytes still describe the same logical values and mapping.
  /// Type ownership and presentation alone do not change a saved format.
  bool HasSameStorage(const RegisterLayout &other) const;

  /// List distinct banks needed to read or write this value. Constants need no
  /// read banks and reject writes. A zero-extending alias may require its
  /// parent bank for a write; unknown or unavailable IDs are errors.
  llvm::Expected<std::vector<RegisterBankID>>
  GetRequiredBanks(RegisterID id, bool writing = false) const;

  /// Assemble and decode a logical value from borrowed bank images. Return an
  /// error for unavailable values, invalid bank extents, missing data, or
  /// unsupported encoding. The returned value owns its bytes.
  llvm::Expected<RegisterValue> Read(RegisterID id,
                                     llvm::ArrayRef<RegisterData> banks) const;

  /// Encode a value into borrowed writable bank images. Preserve unrelated
  /// bytes, apply declared parent zero-extension, and reject nonzero writes to
  /// inactive zero-filled lanes. Constants are read-only; missing banks or
  /// invalid value encodings are errors.
  llvm::Error Write(RegisterID id, const RegisterValue &value,
                    llvm::ArrayRef<MutableRegisterData> banks) const;

private:
  llvm::Error Validate(const RegisterDescription &description) const;
  std::shared_ptr<const RegisterDescription> m_description;
  lldb::ByteOrder m_byte_order;
  std::map<RegisterBankID, RegisterBank> m_banks;
  std::map<RegisterID, RegisterLocation> m_locations;
};

} // namespace lldb_private

#endif
