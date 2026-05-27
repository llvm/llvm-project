//===-- RegisterTypeUnion.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERTYPEUNION_H
#define LLDB_TARGET_REGISTERTYPEUNION_H

#include <stdint.h>
#include <string>
#include <vector>

#include "lldb/Target/RegisterType.h"

namespace lldb_private {

class Stream;
class Log;

class RegisterTypeUnion : public RegisterType {
public:
  typedef std::pair<std::string, const RegisterType *> Field;
  typedef std::vector<Field> Fields;
  RegisterTypeUnion(std::string id, const Fields &fields);

  virtual void ToXMLElement(Stream &strm,
                            const RegisterType *user = nullptr) const override;

  virtual void DumpToLog(Log *log) const override;

  virtual unsigned GetSize() const override;

  const Fields &GetFields() const { return m_fields; }

  static bool classof(const RegisterType *register_type) {
    return register_type->getKind() == RegisterType::eRegisterTypeKindUnion;
  }

  static bool ValidateFields(const Fields &fields);

private:
  Fields m_fields;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERTYPEUNION_H
