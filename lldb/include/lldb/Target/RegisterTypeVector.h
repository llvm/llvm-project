//===-- RegisterTypeVector.h -------------------------------------*- ++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERTYPEVECTOR_H
#define LLDB_TARGET_REGISTERTYPEVECTOR_H

#include <stdint.h>
#include <string>
#include <vector>

#include "lldb/Target/RegisterType.h"
#include "lldb/lldb-enumerations.h"

namespace lldb_private {

class Stream;
class Log;

class RegisterTypeVector : public RegisterType {
public:
  RegisterTypeVector(std::string id, std::string type, unsigned count);

  virtual void ToXMLElement(Stream &strm,
                            const RegisterType *user = nullptr) const override;

  virtual void DumpToLog(Log *log) const override;

  virtual unsigned GetSize() const override;

  static bool classof(const RegisterType *register_type) {
    return register_type->getKind() == RegisterType::eRegisterTypeKindVector;
  }

  const std::string &GetType() const { return m_type; }

  unsigned GetCount() const { return m_count; }

  struct ElementTypeInfo {
    lldb::Encoding encoding = lldb::eEncodingInvalid;
    unsigned size = 0;
  };
  ElementTypeInfo GetElementTypeInfo() const { return m_element_type_info; }

private:
  std::string m_type;
  unsigned m_count;
  ElementTypeInfo m_element_type_info;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERTYPEVECTOR_H
