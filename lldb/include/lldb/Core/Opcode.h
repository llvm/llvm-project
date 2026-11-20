//===-- Opcode.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_OPCODE_H
#define LLDB_CORE_OPCODE_H

#include "lldb/Utility/Endian.h"
#include "lldb/lldb-enumerations.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

namespace lldb {
class SBInstruction;
}

namespace lldb_private {
class DataExtractor;
class Stream;

class Opcode {
public:
  enum Type {
    eTypeInvalid,
    eType8,
    eType16,
    eType16_2,        // a 32-bit Thumb instruction, made up of two words
    eType16_32Tuples, // RISC-V that can have 2, 4, 6, 8 etc byte long
                      // instructions which will be printed in combinations of
                      // 16 & 32-bit words.
    eType32,
    eType64,
    eTypeBytes
  };

  Opcode() = default;

  Opcode(const Opcode &rhs) : m_byte_order(rhs.m_byte_order) { CopyFrom(rhs); }

  Opcode(Opcode &&rhs) noexcept : m_byte_order(rhs.m_byte_order) {
    MoveFrom(std::move(rhs));
  }

  Opcode &operator=(const Opcode &rhs) {
    if (this != &rhs) {
      m_byte_order = rhs.m_byte_order;
      if (HasByteBuf() && rhs.HasByteBuf()) {
        m_data.byte_buf = rhs.m_data.byte_buf;
        m_type = rhs.m_type;
      } else {
        DestroyByteBuf();
        m_type = eTypeInvalid;
        CopyFrom(rhs);
      }
    }
    return *this;
  }

  Opcode &operator=(Opcode &&rhs) noexcept {
    if (this != &rhs) {
      m_byte_order = rhs.m_byte_order;
      if (HasByteBuf() && rhs.HasByteBuf()) {
        m_data.byte_buf = std::move(rhs.m_data.byte_buf);
        m_type = rhs.m_type;
        rhs.Clear();
      } else {
        DestroyByteBuf();
        m_type = eTypeInvalid;
        MoveFrom(std::move(rhs));
      }
    }
    return *this;
  }

  ~Opcode() { DestroyByteBuf(); }

  /// The capacity in bytes of the inline buffer used to store opcode bytes
  /// without dynamic allocation. Instructions <= 16 bytes remain inline,
  /// while longer opcodes spill into heap storage.
  static constexpr size_t kMaxByteSize = 16;

  Opcode(uint8_t inst, lldb::ByteOrder order)
      : m_byte_order(order), m_type(eType8) {
    m_data.inst8 = inst;
  }

  Opcode(uint16_t inst, lldb::ByteOrder order)
      : m_byte_order(order), m_type(eType16) {
    m_data.inst16 = inst;
  }

  Opcode(uint32_t inst, lldb::ByteOrder order)
      : m_byte_order(order), m_type(eType32) {
    m_data.inst32 = inst;
  }

  Opcode(uint64_t inst, lldb::ByteOrder order)
      : m_byte_order(order), m_type(eType64) {
    m_data.inst64 = inst;
  }

  Opcode(uint8_t *bytes, size_t length, Opcode::Type type,
         lldb::ByteOrder order) {
    DoSetOpcodeBytes(bytes, length, type, order);
  }

  void Clear() {
    DestroyByteBuf();
    m_byte_order = lldb::eByteOrderInvalid;
    m_type = Opcode::eTypeInvalid;
  }

  Opcode::Type GetType() const { return m_type; }

  uint8_t GetOpcode8(uint8_t invalid_opcode = UINT8_MAX) const {
    switch (m_type) {
    case Opcode::eTypeInvalid:
      break;
    case Opcode::eType8:
      return m_data.inst8;
    case Opcode::eType16:
      break;
    case Opcode::eType16_2:
      break;
    case Opcode::eType16_32Tuples:
      break;
    case Opcode::eType32:
      break;
    case Opcode::eType64:
      break;
    case Opcode::eTypeBytes:
      break;
    }
    return invalid_opcode;
  }

  uint16_t GetOpcode16(uint16_t invalid_opcode = UINT16_MAX) const {
    switch (m_type) {
    case Opcode::eTypeInvalid:
      break;
    case Opcode::eType8:
      return m_data.inst8;
    case Opcode::eType16:
      return GetEndianSwap() ? llvm::byteswap<uint16_t>(m_data.inst16)
                             : m_data.inst16;
    case Opcode::eType16_2:
      break;
    case Opcode::eType16_32Tuples:
      break;
    case Opcode::eType32:
      break;
    case Opcode::eType64:
      break;
    case Opcode::eTypeBytes:
      break;
    }
    return invalid_opcode;
  }

  uint32_t GetOpcode32(uint32_t invalid_opcode = UINT32_MAX) const {
    switch (m_type) {
    case Opcode::eTypeInvalid:
      break;
    case Opcode::eType8:
      return m_data.inst8;
    case Opcode::eType16:
      return GetEndianSwap() ? llvm::byteswap<uint16_t>(m_data.inst16)
                             : m_data.inst16;
    case Opcode::eType16_32Tuples:
      break;
    case Opcode::eType16_2: // passthrough
    case Opcode::eType32:
      return GetEndianSwap() ? llvm::byteswap<uint32_t>(m_data.inst32)
                             : m_data.inst32;
    case Opcode::eType64:
      break;
    case Opcode::eTypeBytes:
      break;
    }
    return invalid_opcode;
  }

  uint64_t GetOpcode64(uint64_t invalid_opcode = UINT64_MAX) const {
    switch (m_type) {
    case Opcode::eTypeInvalid:
      break;
    case Opcode::eType8:
      return m_data.inst8;
    case Opcode::eType16:
      return GetEndianSwap() ? llvm::byteswap<uint16_t>(m_data.inst16)
                             : m_data.inst16;
    case Opcode::eType16_32Tuples:
      break;
    case Opcode::eType16_2: // passthrough
    case Opcode::eType32:
      return GetEndianSwap() ? llvm::byteswap<uint32_t>(m_data.inst32)
                             : m_data.inst32;
    case Opcode::eType64:
      return GetEndianSwap() ? llvm::byteswap<uint64_t>(m_data.inst64)
                             : m_data.inst64;
    case Opcode::eTypeBytes:
      break;
    }
    return invalid_opcode;
  }

  void SetOpcode8(uint8_t inst, lldb::ByteOrder order) {
    DestroyByteBuf();
    m_type = eType8;
    m_data.inst8 = inst;
    m_byte_order = order;
  }

  void SetOpcode16(uint16_t inst, lldb::ByteOrder order) {
    DestroyByteBuf();
    m_type = eType16;
    m_data.inst16 = inst;
    m_byte_order = order;
  }

  void SetOpcode16_2(uint32_t inst, lldb::ByteOrder order) {
    DestroyByteBuf();
    m_type = eType16_2;
    m_data.inst32 = inst;
    m_byte_order = order;
  }

  void SetOpcode32(uint32_t inst, lldb::ByteOrder order) {
    DestroyByteBuf();
    m_type = eType32;
    m_data.inst32 = inst;
    m_byte_order = order;
  }

  void SetOpcode64(uint64_t inst, lldb::ByteOrder order) {
    DestroyByteBuf();
    m_type = eType64;
    m_data.inst64 = inst;
    m_byte_order = order;
  }

  void SetOpcode16_32TupleBytes(const void *bytes, size_t length,
                                lldb::ByteOrder order) {
    DoSetOpcodeBytes(bytes, length, eType16_32Tuples, order);
  }

  void SetOpcodeBytes(const void *bytes, size_t length) {
    DoSetOpcodeBytes(bytes, length, eTypeBytes, lldb::eByteOrderInvalid);
  }

  void DoSetOpcodeBytes(const void *bytes, size_t length, Opcode::Type type,
                        lldb::ByteOrder order) {
    assert(HasByteBuf(type) && "unexpected Opcode::Type for byte buffer");
    if (bytes != nullptr && length > 0 && HasByteBuf(type)) {
      const auto *src = static_cast<const uint8_t *>(bytes);
      if (HasByteBuf())
        m_data.byte_buf.assign(src, src + length);
      else
        ::new (&m_data.byte_buf) ByteBuf(src, src + length);
      m_type = type;
      m_byte_order = order;
    } else {
      Clear();
    }
  }

  int Dump(Stream *s, uint32_t min_byte_width) const;

  const void *GetOpcodeBytes() const {
    return (HasByteBuf() && !m_data.byte_buf.empty()) ? m_data.byte_buf.data()
                                                      : nullptr;
  }

  uint32_t GetByteSize() const {
    switch (m_type) {
    case Opcode::eTypeInvalid:
      break;
    case Opcode::eType8:
      return sizeof(m_data.inst8);
    case Opcode::eType16:
      return sizeof(m_data.inst16);
    case Opcode::eType16_32Tuples:
      return m_data.byte_buf.size();
    case Opcode::eType16_2: // passthrough
    case Opcode::eType32:
      return sizeof(m_data.inst32);
    case Opcode::eType64:
      return sizeof(m_data.inst64);
    case Opcode::eTypeBytes:
      return m_data.byte_buf.size();
    }
    return 0;
  }

  // Get the opcode exactly as it would be laid out in memory.
  uint32_t GetData(DataExtractor &data) const;

protected:
  friend class lldb::SBInstruction;

  using ByteBuf = llvm::SmallVector<uint8_t, kMaxByteSize>;

  static bool HasByteBuf(Opcode::Type type) {
    return type == eTypeBytes || type == eType16_32Tuples;
  }

  bool HasByteBuf() const { return HasByteBuf(m_type); }

  void DestroyByteBuf() {
    if (HasByteBuf())
      std::destroy_at(&m_data.byte_buf);
  }

  void CopyFrom(const Opcode &rhs) {
    assert(!HasByteBuf());
    switch (rhs.m_type) {
    case eTypeInvalid:
      break;
    case eType8:
      m_data.inst8 = rhs.m_data.inst8;
      break;
    case eType16:
      m_data.inst16 = rhs.m_data.inst16;
      break;
    case eType16_2:
    case eType32:
      m_data.inst32 = rhs.m_data.inst32;
      break;
    case eType64:
      m_data.inst64 = rhs.m_data.inst64;
      break;
    case eType16_32Tuples:
    case eTypeBytes:
      ::new (&m_data.byte_buf) ByteBuf(rhs.m_data.byte_buf);
      break;
    }
    m_type = rhs.m_type;
  }

  void MoveFrom(Opcode &&rhs) noexcept {
    assert(!HasByteBuf());
    switch (rhs.m_type) {
    case eTypeInvalid:
      break;
    case eType8:
      m_data.inst8 = rhs.m_data.inst8;
      break;
    case eType16:
      m_data.inst16 = rhs.m_data.inst16;
      break;
    case eType16_2:
    case eType32:
      m_data.inst32 = rhs.m_data.inst32;
      break;
    case eType64:
      m_data.inst64 = rhs.m_data.inst64;
      break;
    case eType16_32Tuples:
    case eTypeBytes:
      ::new (&m_data.byte_buf) ByteBuf(std::move(rhs.m_data.byte_buf));
      break;
    }
    m_type = rhs.m_type;
    rhs.Clear();
  }

  const void *GetOpcodeDataBytes() const {
    switch (m_type) {
    case Opcode::eTypeInvalid:
      break;
    case Opcode::eType8:
      return &m_data.inst8;
    case Opcode::eType16:
      return &m_data.inst16;
    case Opcode::eType16_32Tuples:
      return m_data.byte_buf.empty() ? nullptr : m_data.byte_buf.data();
    case Opcode::eType16_2: // passthrough
    case Opcode::eType32:
      return &m_data.inst32;
    case Opcode::eType64:
      return &m_data.inst64;
    case Opcode::eTypeBytes:
      return m_data.byte_buf.empty() ? nullptr : m_data.byte_buf.data();
    }
    return nullptr;
  }

  lldb::ByteOrder GetDataByteOrder() const;

  bool GetEndianSwap() const {
    return (m_byte_order == lldb::eByteOrderBig &&
            endian::InlHostByteOrder() == lldb::eByteOrderLittle) ||
           (m_byte_order == lldb::eByteOrderLittle &&
            endian::InlHostByteOrder() == lldb::eByteOrderBig);
  }

  lldb::ByteOrder m_byte_order = lldb::eByteOrderInvalid;

  Opcode::Type m_type = eTypeInvalid;
  union Data {
    // Active member lifetime is managed by Opcode based on m_type.
    Data() {}
    ~Data() {}
    uint8_t inst8;
    uint16_t inst16;
    uint32_t inst32;
    uint64_t inst64;
    ByteBuf byte_buf;
  } m_data;
};

} // namespace lldb_private

#endif // LLDB_CORE_OPCODE_H
