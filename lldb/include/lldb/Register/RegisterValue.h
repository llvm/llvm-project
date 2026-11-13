//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Owned register values and conversions between scalars and encoded bytes.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERVALUE_H
#define LLDB_REGISTER_REGISTERVALUE_H

#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <cstring>
#include <utility>

namespace lldb_private {
class DataExtractor;
class Stream;

/// An owned scalar or byte sequence representing one register value.
///
/// Scalar storage uses host byte order; byte storage retains the byte order
/// supplied by the producer. Memory conversion methods bridge that distinction.
/// This object owns no register identity, target metadata, or process state.
class RegisterValue {
public:
  enum {
    // What we can reasonably put on the stack, big enough to support up to 256
    // byte AArch64 SVE.
    kTypicalRegisterByteSize = 256u,
    // Anything else we'll heap allocate storage for it.
    // 256x256 to support 256 byte AArch64 SME's array storage (ZA) register.
    // Which is a square of vector length x vector length.
    kMaxRegisterByteSize = 256u * 256u,
  };

  typedef llvm::SmallVector<uint8_t, kTypicalRegisterByteSize> BytesContainer;

  enum Type {
    eTypeInvalid,
    eTypeUInt8,
    eTypeUInt16,
    eTypeUInt32,
    eTypeUInt64,
    eTypeUIntN, /// < This value is used when the (integer) register is larger
                /// than 64-bits.
    eTypeFloat,
    eTypeDouble,
    eTypeLongDouble,
    eTypeBytes
  };

  RegisterValue() : m_scalar(static_cast<unsigned long>(0)) {}

  explicit RegisterValue(uint8_t inst) : m_type(eTypeUInt8) { m_scalar = inst; }

  explicit RegisterValue(uint16_t inst) : m_type(eTypeUInt16) {
    m_scalar = inst;
  }

  explicit RegisterValue(uint32_t inst) : m_type(eTypeUInt32) {
    m_scalar = inst;
  }

  explicit RegisterValue(uint64_t inst) : m_type(eTypeUInt64) {
    m_scalar = inst;
  }

  explicit RegisterValue(llvm::APInt inst) : m_type(eTypeUIntN) {
    m_scalar = llvm::APInt(std::move(inst));
  }

  explicit RegisterValue(float value) : m_type(eTypeFloat) { m_scalar = value; }

  explicit RegisterValue(double value) : m_type(eTypeDouble) {
    m_scalar = value;
  }

  explicit RegisterValue(long double value) : m_type(eTypeLongDouble) {
    m_scalar = value;
  }

  explicit RegisterValue(llvm::ArrayRef<uint8_t> bytes,
                         lldb::ByteOrder byte_order) {
    SetBytes(bytes.data(), bytes.size(), byte_order);
  }

  RegisterValue::Type GetType() const { return m_type; }

  /// Copy the representation and owned bytes; return false for an invalid
  /// source.
  bool CopyValue(const RegisterValue &rhs);

  /// Change the representation tag without converting the existing value.
  void SetType(RegisterValue::Type type) { m_type = type; }

  /// Select storage from the register encoding and width without populating a
  /// value. Return eTypeInvalid when the encoding or width is unsupported.
  RegisterValue::Type SetType(lldb::Encoding encoding, uint32_t byte_size);

  /// Expose a borrowed view of the current representation and its byte order.
  /// The view must not outlive this value or a mutation of its storage. Return
  /// false when the representation has no data.
  bool GetData(DataExtractor &data) const;

  /// Create an owning extractor for the least significant requested bytes,
  /// converted to the requested byte order. Return false when the value is too
  /// short or either byte order is invalid.
  bool GetData(DataExtractor &data, uint32_t byte_size,
               lldb::ByteOrder byte_order) const;

  /// Copy a register-width value to a buffer in the requested byte order. Copy
  /// its least significant bytes when the destination is shorter, or pad with
  /// zeroes when it is longer. Return the bytes copied, or zero on error.
  uint32_t GetAsMemoryData(uint32_t byte_size, void *dst, uint32_t dst_len,
                           lldb::ByteOrder dst_byte_order, Status &error,
                           llvm::StringRef name = {}) const;

  /// Decode source bytes using the register encoding and declared width. Short
  /// inputs are zero-padded in the supplied byte order; oversized inputs are
  /// rejected. Return the number of source bytes consumed, or zero on error.
  uint32_t SetFromMemoryData(lldb::Encoding encoding, uint32_t byte_size,
                             const void *src, uint32_t src_len,
                             lldb::ByteOrder src_byte_order, Status &error,
                             llvm::StringRef name = {});

  /// Copy a scalar interpretation when supported by the current representation.
  /// Return false when the value cannot be represented as a Scalar.
  bool GetScalarValue(Scalar &scalar) const;

  uint8_t GetAsUInt8(uint8_t fail_value = UINT8_MAX,
                     bool *success_ptr = nullptr) const {
    if (m_type == eTypeUInt8) {
      if (success_ptr)
        *success_ptr = true;
      return m_scalar.UChar(fail_value);
    }
    if (success_ptr)
      *success_ptr = true;
    return fail_value;
  }

  uint16_t GetAsUInt16(uint16_t fail_value = UINT16_MAX,
                       bool *success_ptr = nullptr) const;

  uint32_t GetAsUInt32(uint32_t fail_value = UINT32_MAX,
                       bool *success_ptr = nullptr) const;

  uint64_t GetAsUInt64(uint64_t fail_value = UINT64_MAX,
                       bool *success_ptr = nullptr) const;

  llvm::APInt GetAsUInt128(const llvm::APInt &fail_value,
                           bool *success_ptr = nullptr) const;

  float GetAsFloat(float fail_value = 0.0f, bool *success_ptr = nullptr) const;

  double GetAsDouble(double fail_value = 0.0,
                     bool *success_ptr = nullptr) const;

  long double GetAsLongDouble(long double fail_value = 0.0,
                              bool *success_ptr = nullptr) const;

  void SetValueToInvalid() { m_type = eTypeInvalid; }

  /// Clear a bit of a supported integer or byte representation. Return false
  /// when the bit or representation is unsupported.
  bool ClearBit(uint32_t bit);

  /// Set a bit of a supported integer or byte representation. Return false when
  /// the bit or representation is unsupported.
  bool SetBit(uint32_t bit);

  bool operator==(const RegisterValue &rhs) const;

  bool operator!=(const RegisterValue &rhs) const;

  void operator=(uint8_t uint) {
    m_type = eTypeUInt8;
    m_scalar = uint;
  }

  void operator=(uint16_t uint) {
    m_type = eTypeUInt16;
    m_scalar = uint;
  }

  void operator=(uint32_t uint) {
    m_type = eTypeUInt32;
    m_scalar = uint;
  }

  void operator=(uint64_t uint) {
    m_type = eTypeUInt64;
    m_scalar = uint;
  }

  void operator=(llvm::APInt uint) {
    m_type = eTypeUIntN;
    m_scalar = llvm::APInt(std::move(uint));
  }

  void operator=(float f) {
    m_type = eTypeFloat;
    m_scalar = f;
  }

  void operator=(double f) {
    m_type = eTypeDouble;
    m_scalar = f;
  }

  void operator=(long double f) {
    m_type = eTypeLongDouble;
    m_scalar = f;
  }

  void SetUInt8(uint8_t uint) {
    m_type = eTypeUInt8;
    m_scalar = uint;
  }

  void SetUInt16(uint16_t uint) {
    m_type = eTypeUInt16;
    m_scalar = uint;
  }

  void SetUInt32(uint32_t uint, Type t = eTypeUInt32) {
    m_type = t;
    m_scalar = uint;
  }

  void SetUInt64(uint64_t uint, Type t = eTypeUInt64) {
    m_type = t;
    m_scalar = uint;
  }

  void SetUIntN(llvm::APInt uint) {
    m_type = eTypeUIntN;
    m_scalar = std::move(uint);
  }

  /// Choose integer storage large enough for the requested byte width. A zero
  /// width selects 64-bit storage; widths above 16 bytes are unsupported.
  bool SetUInt(uint64_t uint, uint32_t byte_size);

  void SetFloat(float f) {
    m_type = eTypeFloat;
    m_scalar = f;
  }

  void SetDouble(double f) {
    m_type = eTypeDouble;
    m_scalar = f;
  }

  void SetLongDouble(long double f) {
    m_type = eTypeLongDouble;
    m_scalar = f;
  }

  /// Copy bytes and retain their byte order. Null or empty input invalidates
  /// the value; the caller can release the source buffer after this call.
  void SetBytes(const void *bytes, size_t length, lldb::ByteOrder byte_order);

  /// Extend the selected sign bit through integer storage. Return false for
  /// non-integer representations or an invalid sign position.
  bool SignExtend(uint32_t sign_bitpos);

  /// Parse a scalar or vector value using the register encoding and width.
  /// Return an error for malformed input or a value that does not fit.
  Status SetValueFromString(lldb::Encoding encoding, uint32_t byte_size,
                            llvm::StringRef value_str);
  Status SetValueFromString(lldb::Encoding encoding, uint32_t byte_size,
                            const char *value_str) = delete;

  /// Decode a register at the supplied offset without changing the extractor.
  /// When partial data is allowed, missing bytes are zero-padded; otherwise a
  /// short input is an error. The register encoding determines the
  /// representation.
  Status SetValueFromData(lldb::Encoding encoding, uint32_t byte_size,
                          DataExtractor &data, lldb::offset_t offset,
                          bool partial_data_ok, llvm::StringRef name = {});

  /// Return borrowed representation bytes, or nullptr for an invalid value.
  /// Interpret these bytes using GetByteOrder(), not necessarily target byte
  /// order.
  const void *GetBytes() const;

  lldb::ByteOrder GetByteOrder() const {
    if (m_type == eTypeBytes)
      return buffer.byte_order;
    return endian::InlHostByteOrder();
  }

  uint32_t GetByteSize() const;

  /// Discard the current value and mark it invalid.
  void Clear();

protected:
  RegisterValue::Type m_type = eTypeInvalid;
  Scalar m_scalar;

  struct RegisterValueBuffer {
    // Start at max stack storage size. Move to the heap for anything larger.
    RegisterValueBuffer() : bytes(kTypicalRegisterByteSize) {}

    mutable BytesContainer bytes;
    lldb::ByteOrder byte_order = lldb::eByteOrderInvalid;
  } buffer;
};

} // namespace lldb_private

#endif // LLDB_REGISTER_REGISTERVALUE_H
