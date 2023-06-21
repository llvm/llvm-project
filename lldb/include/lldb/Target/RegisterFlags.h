//===-- RegisterFlags.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERFLAGS_H
#define LLDB_TARGET_REGISTERFLAGS_H

#include "lldb/Utility/Log.h"

namespace lldb_private {

class RegisterFlags {
public:
  class Field {
  public:
    Field(std::string name, unsigned start, unsigned end)
        : m_name(std::move(name)), m_start(start), m_end(end) {
      assert(m_start <= m_end && "Start bit must be <= end bit.");
    }

    /// Get size of the field in bits. Will always be at least 1.
    unsigned GetSizeInBits() const { return m_end - m_start + 1; }

    /// A mask that covers all bits of the field.
    uint64_t GetMask() const {
      return (((uint64_t)1 << (GetSizeInBits())) - 1) << m_start;
    }

    /// Extract value of the field from a whole register value.
    uint64_t GetValue(uint64_t register_value) const {
      return (register_value & GetMask()) >> m_start;
    }

    const std::string &GetName() const { return m_name; }
    unsigned GetStart() const { return m_start; }
    unsigned GetEnd() const { return m_end; }
    bool Overlaps(const Field &other) const;
    void log(Log *log) const;

    /// Return the number of bits between this field and the other, that are not
    /// covered by either field.
    unsigned PaddingDistance(const Field &other) const;

    bool operator<(const Field &rhs) const {
      return GetStart() < rhs.GetStart();
    }

    bool operator==(const Field &rhs) const {
      return (m_name == rhs.m_name) && (m_start == rhs.m_start) &&
             (m_end == rhs.m_end);
    }

  private:
    std::string m_name;
    /// Start/end bit positions. Where start N, end N means a single bit
    /// field at position N. We expect that start <= end. Bit positions begin
    /// at 0.
    /// Start is the LSB, end is the MSB.
    unsigned m_start;
    unsigned m_end;
  };

  /// This assumes that:
  /// * There is at least one field.
  /// * The fields are sorted in descending order.
  /// Gaps are allowed, they will be filled with anonymous padding fields.
  RegisterFlags(std::string id, unsigned size,
                const std::vector<Field> &fields);

  // Reverse the order of the fields, keeping their values the same.
  // For example a field from bit 31 to 30 with value 0b10 will become bits
  // 1 to 0, with the same 0b10 value.
  // Use this when you are going to show the register using a bitfield struct
  // type. If that struct expects MSB first and you are on little endian where
  // LSB would be first, this corrects that (and vice versa for big endian).
  template <typename T> T ReverseFieldOrder(T value) const {
    T ret = 0;
    unsigned shift = 0;
    for (auto field : GetFields()) {
      ret |= field.GetValue(value) << shift;
      shift += field.GetSizeInBits();
    }

    return ret;
  }

  const std::vector<Field> &GetFields() const { return m_fields; }
  const std::string &GetID() const { return m_id; }
  unsigned GetSize() const { return m_size; }
  void log(Log *log) const;

  /// Produce a text table showing the layout of all the fields. Unnamed/padding
  /// fields will be included, with only their positions shown.
  /// max_width will be the width in characters of the terminal you are
  /// going to print the table to. If the table would exceed this width, it will
  /// be split into many tables as needed.
  std::string AsTable(uint32_t max_width) const;

private:
  const std::string m_id;
  /// Size in bytes
  const unsigned m_size;
  std::vector<Field> m_fields;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERFLAGS_H
