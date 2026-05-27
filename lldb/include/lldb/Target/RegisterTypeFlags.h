//===-- RegisterTypeFlags.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERTYPEFLAGS_H
#define LLDB_TARGET_REGISTERTYPEFLAGS_H

#include <stdint.h>
#include <string>
#include <vector>

#include "lldb/Target/RegisterType.h"
#include "llvm/ADT/StringSet.h"

namespace lldb_private {

class Stream;
class Log;

class RegisterTypeEnum : public RegisterType {
public:
  struct Enumerator {
    uint64_t m_value;
    // Short name for the value. Shown in tables and when printing the field's
    // value. For example "RZ".
    std::string m_name;

    Enumerator(uint64_t value, std::string name)
        : m_value(value), m_name(std::move(name)) {}

    void DumpToLog(Log *log) const;

    void ToXMLElement(Stream &strm) const;
  };

  typedef std::vector<Enumerator> Enumerators;

  // GDB also includes a "size" that is the size of the underlying register.
  // We will not store that here but instead use the size of the register
  // this gets attached to when emitting XML.
  RegisterTypeEnum(std::string id, const Enumerators &enumerators);

  const Enumerators &GetEnumerators() const { return m_enumerators; }

  virtual void DumpToLog(Log *log) const override;

  virtual unsigned GetSize() const override {
    // Enums don't have a size until they are used by a specific register,
    // so we return 0 just to be sure they don't end up attached directly to a
    // register. We expect them to only be used by flags, then the flags are
    // attached to the register.
    return 0;
  }

  virtual void ToXMLElement(Stream &strm,
                            const RegisterType *user = nullptr) const override;

  static bool classof(const RegisterType *register_type) {
    return register_type->getKind() == RegisterType::eRegisterTypeKindEnum;
  }

private:
  Enumerators m_enumerators;
};

class RegisterTypeFlags : public RegisterType {
public:
  class Field {
  public:
    /// Where start is the least significant bit and end is the most
    /// significant bit. The start bit must be <= the end bit.
    Field(std::string name, unsigned start, unsigned end);

    /// Construct a field that also has some known enum values.
    Field(std::string name, unsigned start, unsigned end,
          const RegisterTypeEnum *enum_type);

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

    const std::string &GetName() const { return m_name; }
    unsigned GetStart() const { return m_start; }
    unsigned GetEnd() const { return m_end; }
    const RegisterTypeEnum *GetEnum() const { return m_enum_type; }
    bool Overlaps(const Field &other) const;
    void DumpToLog(Log *log) const;

    /// Return the number of bits between this field and the other, that are not
    /// covered by either field.
    unsigned PaddingDistance(const Field &other) const;

    void ToXMLElement(Stream &strm) const;

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

    const RegisterTypeEnum *m_enum_type;
  };

  /// This assumes that:
  /// * There is at least one field.
  /// * The fields are sorted in descending order.
  /// Gaps are allowed.
  RegisterTypeFlags(std::string id, unsigned size,
                    const std::vector<Field> &fields);

  /// Replace all the fields with the new set of fields. All the assumptions
  /// and checks apply as when you use the constructor. Intended to only be used
  /// when runtime field detection is needed.
  void SetFields(const std::vector<Field> &fields);

  /// Make a string where each line contains the name of a field that has
  /// enum values, and lists what those values are.
  std::string DumpEnums(uint32_t max_width) const;

  const std::vector<Field> &GetFields() const { return m_fields; }
  virtual unsigned GetSize() const override { return m_size; }

  virtual void DumpToLog(Log *log) const override;

  /// Produce a text table showing the layout of all the fields. Unnamed/padding
  /// fields will be included, with only their positions shown.
  /// max_width will be the width in characters of the terminal you are
  /// going to print the table to. If the table would exceed this width, it will
  /// be split into many tables as needed.
  std::string AsTable(uint32_t max_width) const;

  virtual void ToXMLElement(Stream &strm,
                            const RegisterType *user = nullptr) const override;

  static bool classof(const RegisterType *register_type) {
    return register_type->getKind() == RegisterType::eRegisterTypeKindFlags;
  }

private:
  /// Size in bytes
  const unsigned m_size;
  std::vector<Field> m_fields;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERTYPEFLAGS_H
