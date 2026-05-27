//===-- RegisterType.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERTYPE_H
#define LLDB_TARGET_REGISTERTYPE_H

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
    eRegisterTypeKindUnion,
    eRegisterTypeKindVector,
  };

  RegisterTypeKind getKind() const { return m_kind; }

  RegisterType(RegisterTypeKind kind, std::string id)
      : m_kind(kind), m_id(std::move(id)) {}

  /// Output XML that describes this type, to be inserted into a target XML
  /// file. Reserved characters like "<" are replaced with their XML safe
  /// equivalents like "&gt;".
  void ToXML(Stream &strm,
             std::unordered_set<const RegisterType *> &previously_emitted,
             const RegisterType *user = nullptr) const;

  virtual ~RegisterType() = default;

  /// Output the register type as an XML element. That is, "<foo ...>" until the
  /// closing </foo>, including any child types in between. For example the
  /// flags in a register flag set.
  virtual void ToXMLElement(Stream &strm,
                            const RegisterType *user = nullptr) const = 0;

  const std::string &GetID() const { return m_id; }

  void SetDependencies(const std::vector<const RegisterType *> dependencies) {
    m_dependencies = dependencies;
  }

  virtual void DumpToLog(Log *log) const = 0;

  /// The size of the type in bytes. Return 0 if the size is unknown or context
  /// specific.
  virtual unsigned GetSize() const = 0;

private:
  const RegisterTypeKind m_kind;
  const std::string m_id;
  std::vector<const RegisterType *> m_dependencies;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERTYPE_H
