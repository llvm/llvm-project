//===-- SBSymbol.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbol_h_
#define LLDB_SBSymbol_h_

#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBTarget.h"

namespace lldb {

class LLDB_API SBSymbol {
public:
  SBSymbol();

  ~SBSymbol();

  SBSymbol(const lldb::SBSymbol &rhs);

  const lldb::SBSymbol &operator=(const lldb::SBSymbol &rhs);

  bool IsValid() const;

  const char *GetName() const;

  const char *GetDisplayName() const;

  const char *GetMangledName() const;

  lldb::SBInstructionList GetInstructions(lldb::SBTarget target);

  lldb::SBInstructionList GetInstructions(lldb::SBTarget target,
                                          const char *flavor_string);

  SBAddress GetStartAddress();

  SBAddress GetEndAddress();

  uint32_t GetPrologueByteSize();

  SymbolType GetType();

  bool operator==(const lldb::SBSymbol &rhs) const;

  bool operator!=(const lldb::SBSymbol &rhs) const;

  bool GetDescription(lldb::SBStream &description);

  //----------------------------------------------------------------------
  // Returns true if the symbol is externally visible in the module that it is
  // defined in
  //----------------------------------------------------------------------
  bool IsExternal();

  //----------------------------------------------------------------------
  // Returns true if the symbol was synthetically generated from something
  // other than the actual symbol table itself in the object file.
  //----------------------------------------------------------------------
  bool IsSynthetic();

protected:
  lldb_private::Symbol *get();

  void reset(lldb_private::Symbol *);

private:
  friend class SBAddress;
  friend class SBFrame;
  friend class SBModule;
  friend class SBSymbolContext;

  SBSymbol(lldb_private::Symbol *lldb_object_ptr);

  void SetSymbol(lldb_private::Symbol *lldb_object_ptr);

  lldb_private::Symbol *m_opaque_ptr;
};

} // namespace lldb

#endif // LLDB_SBSymbol_h_
