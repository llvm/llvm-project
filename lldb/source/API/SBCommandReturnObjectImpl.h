//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_API_SBCOMMANDRETURNOBJECTIMPL_H
#define LLDB_SOURCE_API_SBCOMMANDRETURNOBJECTIMPL_H

namespace lldb_private {
class CommandReturnObject;

class SBCommandReturnObjectImpl {
public:
  SBCommandReturnObjectImpl();
  SBCommandReturnObjectImpl(CommandReturnObject &ref);
  SBCommandReturnObjectImpl(const SBCommandReturnObjectImpl &rhs);
  SBCommandReturnObjectImpl &operator=(const SBCommandReturnObjectImpl &rhs);
  ~SBCommandReturnObjectImpl();

  CommandReturnObject *get() const { return m_ptr; }

private:
  CommandReturnObject *m_ptr;
  bool m_owned = true;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_API_SBCOMMANDRETURNOBJECTIMPL_H
