//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REGISTERTYPE_H
#define LLDB_UTILITY_REGISTERTYPE_H

#include "lldb/lldb-enumerations.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace lldb_private {

class Stream;
class Log;

class RegisterType {
public:
  enum RegisterTypeKind {
    eRegisterTypeKindFlags,
    eRegisterTypeKindEnum,
    eRegisterTypeKindBuiltin,
    eRegisterTypeKindVector,
  };

  RegisterTypeKind getKind() const { return m_kind; }

  RegisterType(RegisterTypeKind kind, std::string id);
  RegisterType(const RegisterType &) = delete;
  RegisterType &operator=(const RegisterType &) = delete;
  RegisterType(RegisterType &&) = delete;
  RegisterType &operator=(RegisterType &&) = delete;

  /// Output XML that describes this type, to be inserted into a target XML
  /// file. Reserved characters like "<" are replaced with their XML safe
  /// equivalents like "&gt;".
  void ToXML(Stream &strm, std::unordered_set<std::string> &previously_emitted,
             const RegisterType *user = nullptr) const;

  virtual ~RegisterType() = default;

  /// Output the register type as an XML element. That is, "<foo ...>" until the
  /// closing </foo>, including any child types in between. For example the
  /// flags in a register flag set.
  virtual void ToXMLElement(Stream &strm,
                            const RegisterType *user = nullptr) const = 0;

  const std::string &GetID() const { return m_id; }

  /// Return an identifier unique among all RegisterType instances constructed
  /// during the lifetime of the LLDB host process. The identifier is not
  /// reused after this instance is destroyed.
  uint64_t GetUID() const { return m_uid; }

  /// Return this type's size in bytes when the target description defines it.
  virtual std::optional<uint64_t> GetByteSize() const { return std::nullopt; }

  void SetDependencies(std::vector<const RegisterType *> dependencies) {
    m_dependencies = dependencies;
  }

private:
  const RegisterTypeKind m_kind;
  const std::string m_id;
  const uint64_t m_uid;
  std::vector<const RegisterType *> m_dependencies;
};

/// A predefined GDB target-description type.
class RegisterTypeBuiltin : public RegisterType {
public:
  /// A byte size of zero means the size is target-dependent.
  RegisterTypeBuiltin(std::string id, lldb::Encoding encoding,
                      lldb::Format format, uint32_t byte_size);

  lldb::Encoding GetEncoding() const { return m_encoding; }
  lldb::Format GetFormat() const { return m_format; }
  std::optional<uint64_t> GetByteSize() const override {
    return m_byte_size ? std::optional<uint64_t>(m_byte_size) : std::nullopt;
  }

  void ToXMLElement(Stream &strm,
                    const RegisterType *user = nullptr) const override;

  static bool classof(const RegisterType *type) {
    return type->getKind() == eRegisterTypeKindBuiltin;
  }

private:
  lldb::Encoding m_encoding;
  lldb::Format m_format;
  uint32_t m_byte_size;
};

/// A GDB target-description vector type. The element type must outlive it.
class RegisterTypeVector : public RegisterType {
public:
  RegisterTypeVector(std::string id, const RegisterType *element_type,
                     uint32_t count);

  const RegisterType *GetElementType() const { return m_element_type; }
  uint32_t GetCount() const { return m_count; }
  std::optional<uint64_t> GetByteSize() const override;

  void ToXMLElement(Stream &strm,
                    const RegisterType *user = nullptr) const override;

  static bool classof(const RegisterType *type) {
    return type->getKind() == eRegisterTypeKindVector;
  }

private:
  const RegisterType *m_element_type;
  uint32_t m_count;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_REGISTERTYPE_H
