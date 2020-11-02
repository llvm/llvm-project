//===-- UUID.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_UUID_H
#define LLDB_UTILITY_UUID_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include <stddef.h>
#include <stdint.h>
#include <string>

namespace lldb_private {

  class Stream;

class UUID {
public:
  UUID() = default;

  // Reference:
  // https://crashpad.chromium.org/doxygen/structcrashpad_1_1CodeViewRecordPDB70.html
  struct CvRecordPdb70 {
    struct {
      llvm::support::ulittle32_t Data1;
      llvm::support::ulittle16_t Data2;
      llvm::support::ulittle16_t Data3;
      uint8_t Data4[8];
    } Uuid;
    llvm::support::ulittle32_t Age;
    // char PDBFileName[];
  };

  /// Create a UUID from CvRecordPdb70.
  static UUID fromCvRecord(CvRecordPdb70 debug_info);

  /// Creates a UUID from the data pointed to by the bytes argument. No special
  /// significance is attached to any of the values.
  static UUID fromData(const void *bytes, uint32_t num_bytes) {
    if (bytes)
      return fromData({reinterpret_cast<const uint8_t *>(bytes), num_bytes});
    return UUID();
  }

  /// Creates a uuid from the data pointed to by the bytes argument. No special
  /// significance is attached to any of the values.
  static UUID fromData(llvm::ArrayRef<uint8_t> bytes) { return UUID(bytes); }

  /// Creates a UUID from the data pointed to by the bytes argument. Data
  /// consisting purely of zero bytes is treated as an invalid UUID.
  static UUID fromOptionalData(const void *bytes, uint32_t num_bytes) {
    if (bytes)
      return fromOptionalData(
          {reinterpret_cast<const uint8_t *>(bytes), num_bytes});
    return UUID();
  }

  /// Creates a UUID from the data pointed to by the bytes argument. Data
  /// consisting purely of zero bytes is treated as an invalid UUID.
  static UUID fromOptionalData(llvm::ArrayRef<uint8_t> bytes) {
    if (llvm::all_of(bytes, [](uint8_t b) { return b == 0; }))
      return UUID();
    return UUID(bytes);
  }

  void Clear() { m_bytes.clear(); }

  void Dump(Stream *s) const;

  llvm::ArrayRef<uint8_t> GetBytes() const { return m_bytes; }

  explicit operator bool() const { return IsValid(); }
  bool IsValid() const { return !m_bytes.empty(); }

  std::string GetAsString(llvm::StringRef separator = "-") const;

  bool SetFromStringRef(llvm::StringRef str);

  // Same as SetFromStringRef, but if the resultant UUID is all 0 bytes, set the
  // UUID to invalid.
  bool SetFromOptionalStringRef(llvm::StringRef str);

  /// Decode as many UUID bytes as possible from the C string \a cstr.
  ///
  /// \param[in] str
  ///     An llvm::StringRef that points at a UUID string value (no leading
  ///     spaces). The string must contain only hex characters and optionally
  ///     can contain the '-' sepearators.
  ///
  /// \param[in] uuid_bytes
  ///     A buffer of bytes that will contain a full or partially decoded UUID.
  ///
  /// \return
  ///     The original string, with all decoded bytes removed.
  static llvm::StringRef
  DecodeUUIDBytesFromString(llvm::StringRef str,
                            llvm::SmallVectorImpl<uint8_t> &uuid_bytes);

private:
  UUID(llvm::ArrayRef<uint8_t> bytes) : m_bytes(bytes.begin(), bytes.end()) {}

  // GNU ld generates 20-byte build-ids. Size chosen to avoid heap allocations
  // for this case.
  llvm::SmallVector<uint8_t, 20> m_bytes;

  friend bool operator==(const UUID &LHS, const UUID &RHS) {
    return LHS.m_bytes == RHS.m_bytes;
  }
  friend bool operator!=(const UUID &LHS, const UUID &RHS) {
    return !(LHS == RHS);
  }
  friend bool operator<(const UUID &LHS, const UUID &RHS) {
    return LHS.m_bytes < RHS.m_bytes;
  }
  friend bool operator<=(const UUID &LHS, const UUID &RHS) {
    return !(RHS < LHS);
  }
  friend bool operator>(const UUID &LHS, const UUID &RHS) { return RHS < LHS; }
  friend bool operator>=(const UUID &LHS, const UUID &RHS) {
    return !(LHS < RHS);
  }
};
} // namespace lldb_private

#endif // LLDB_UTILITY_UUID_H
