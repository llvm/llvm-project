//===-- ImportedDeclaration.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_IMPORTED_DECLARATION_H
#define LLDB_SYMBOL_IMPORTED_DECLARATION_H

#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/UserID.h"

namespace lldb_private {

struct ImportedDeclaration : public UserID {

  ImportedDeclaration(lldb::user_id_t uid, ConstString name,
                      SymbolFile *symbol_file)
      : UserID(uid), m_name(name), m_symbol_file(symbol_file) {}

  ConstString GetName() const { return m_name; }

  std::vector<lldb_private::CompilerContext> GetDeclContext() const;

private:
  ConstString m_name;
  SymbolFile *m_symbol_file = nullptr;
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_IMPORTED_DECLARATION_H
