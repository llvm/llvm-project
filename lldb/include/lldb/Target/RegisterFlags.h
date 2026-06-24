//===-- RegisterFlags.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERFLAGS_H
#define LLDB_TARGET_REGISTERFLAGS_H

#include "lldb/lldb-enumerations.h"

#include <stdint.h>
#include <string>
#include <vector>

#include "llvm/ADT/StringSet.h"

namespace lldb_private {

class Stream;
class Log;

class FieldEnum {
public:
  struct Enumerator {
    uint64_t m_value;
    // Short name for the value. Shown in tables and when printing the field's
    // value. For example "RZ".
    std::string m_name;

    Enumerator(uint64_t value, std::string name)
        : m_value(value), m_name(std::move(name)) {}

    void ToXML(Stream &strm) const;

    void DumpToLog(Log *log) const;
  };

  typedef std::vector<Enumerator> Enumerators;

  // GDB also includes a "size" that is the size of the underlying register.
  // We will not store that here but instead use the size of the register
  // this gets attached to when emitting XML.
  FieldEnum(std::string id, const Enumerators &enumerators);

  const Enumerators &GetEnumerators() const { return m_enumerators; }

  const std::string &GetID() const { return m_id; }

  void ToXML(Stream &strm, unsigned size) const;

  void DumpToLog(Log *log) const;

private:
  std::string m_id;
  Enumerators m_enumerators;
};

class RegisterFlags {
public:
  class Field {
  public:
    /// Where start is the least significant bit and end is the most
    /// significant bit. The start bit must be <= the end bit.
    Field(std::string name, unsigned start, unsigned end);

    /// Construct a field that also has some known enum values.
    Field(std::string name, unsigned start, unsigned end,
          const FieldEnum *enum_type);

    /// Construct a field that occupies a single bit.
    Field(std::string name, unsigned bit_position);

    /// Get size of the field in bits. Will always be at least 1.
    unsigned GetSizeInBits() const;

    /// Identical to GetSizeInBits, but for the GDB client to use.
    static unsigned GetSizeInBits(unsigned start, unsigned end);

    /// A mask that covers all bits of the field.
    uint64_t GetMask() const;

    /// The maximum unsigned value that could be contained in this field.
    uint64_t GetMaxValue() const;

    /// Identical to GetMaxValue but for the GDB client to use.
    static uint64_t GetMaxValue(unsigned start, unsigned end);

    /// Extract value of the field from a whole register value.
    uint64_t GetValue(uint64_t register_value) const {
      return (register_value & GetMask()) >> m_start;
    }

    const std::string &GetName() const { return m_name; }
    unsigned GetStart() const { return m_start; }
    unsigned GetEnd() const { return m_end; }
    const FieldEnum *GetEnum() const { return m_enum_type; }
    bool Overlaps(const Field &other) const;
    void DumpToLog(Log *log) const;

    /// Return the number of bits between this field and the other, that are not
    /// covered by either field.
    unsigned PaddingDistance(const Field &other) const;

    /// Output XML that describes this field, to be inserted into a target XML
    /// file. Reserved characters in field names like "<" are replaced with
    /// their XML safe equivalents like "&gt;".
    void ToXML(Stream &strm) const;

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

    const FieldEnum *m_enum_type;
  };

  /// This assumes that:
  /// * There is at least one field.
  /// * The fields are sorted in descending order.
  /// Gaps are allowed, they will be filled with anonymous padding fields.
  RegisterFlags(std::string id, unsigned size,
                const std::vector<Field> &fields);

  /// Replace all the fields with the new set of fields. All the assumptions
  /// and checks apply as when you use the constructor. Intended to only be used
  /// when runtime field detection is needed.
  void SetFields(const std::vector<Field> &fields);

  /// Make a string where each line contains the name of a field that has
  /// enum values, and lists what those values are.
  std::string DumpEnums(uint32_t max_width) const;

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
  void DumpToLog(Log *log) const;

  /// Produce a text table showing the layout of all the fields. Unnamed/padding
  /// fields will be included, with only their positions shown.
  /// max_width will be the width in characters of the terminal you are
  /// going to print the table to. If the table would exceed this width, it will
  /// be split into many tables as needed.
  std::string AsTable(uint32_t max_width) const;

  /// Output XML that describes this set of flags.
  /// EnumsToXML should have been called before this.
  void ToXML(Stream &strm) const;

  /// Enum types must be defined before use, and
  /// GDBRemoteCommunicationServerLLGS view of the register types is based only
  /// on the registers. So this method emits any enum types that the upcoming
  /// set of fields may need. "seen" is a set of Enum IDs that we have already
  /// printed, that is updated with any printed by this call. This prevents us
  /// printing the same enum multiple times.
  void EnumsToXML(Stream &strm, llvm::StringSet<> &seen) const;

private:
  const std::string m_id;
  /// Size in bytes
  const unsigned m_size;
  std::vector<Field> m_fields;
};

/// Represents a union type from a GDB remote target description XML. Each
/// field is an alternative interpretation of the same register data (e.g.,
/// viewing an FPU register as both ieee_single and ieee_double).
class RegisterUnion {
public:
  class Field {
  public:
    /// Construct a union field. For scalar fields, \p vector_count should be 0.
    /// For vector fields, \p byte_size is the element size and \p vector_count
    /// is the number of elements.
    Field(std::string name, lldb::Encoding encoding, lldb::Format format,
          uint32_t byte_size, uint32_t vector_count = 0);

    const std::string &GetName() const { return m_name; }
    lldb::Encoding GetEncoding() const { return m_encoding; }
    lldb::Format GetFormat() const { return m_format; }
    uint32_t GetByteSize() const { return m_byte_size; }
    uint32_t GetVectorCount() const { return m_vector_count; }
    bool IsVector() const { return m_vector_count > 0; }
    /// For scalar fields, returns byte_size. For vector fields, returns
    /// element byte_size * vector_count.
    uint32_t GetTotalByteSize() const {
      if (m_vector_count > 0) {
        uint64_t total = static_cast<uint64_t>(m_byte_size) * m_vector_count;
        assert(total <= UINT32_MAX && "Vector type size overflow");
        return static_cast<uint32_t>(total);
      }
      return m_byte_size;
    }
    void DumpToLog(Log *log) const;

  private:
    std::string m_name;
    lldb::Encoding m_encoding;
    lldb::Format m_format;
    uint32_t m_byte_size;
    uint32_t m_vector_count;
  };

  /// \p fields must not be empty.
  RegisterUnion(std::string id, std::vector<Field> fields);

  const std::vector<Field> &GetFields() const { return m_fields; }
  const std::string &GetID() const { return m_id; }
  /// Returns the size of the largest field in bytes.
  uint32_t GetSize() const { return m_size; }
  void DumpToLog(Log *log) const;

private:
  const std::string m_id;
  uint32_t m_size;
  std::vector<Field> m_fields;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERFLAGS_H
