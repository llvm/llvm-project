//===-- SBInstructionList.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBInstructionList_h_
#define LLDB_SBInstructionList_h_

#include "lldb/API/SBDefines.h"

#include <stdio.h>

namespace lldb {

class LLDB_API SBInstructionList {
public:
  SBInstructionList();

  SBInstructionList(const SBInstructionList &rhs);

  const SBInstructionList &operator=(const SBInstructionList &rhs);

  ~SBInstructionList();

  bool IsValid() const;

  size_t GetSize();

  lldb::SBInstruction GetInstructionAtIndex(uint32_t idx);

  void Clear();

  void AppendInstruction(lldb::SBInstruction inst);

  void Print(FILE *out);

  bool GetDescription(lldb::SBStream &description);

  bool DumpEmulationForAllInstructions(const char *triple);

protected:
  friend class SBFunction;
  friend class SBSymbol;
  friend class SBTarget;

  void SetDisassembler(const lldb::DisassemblerSP &opaque_sp);

private:
  lldb::DisassemblerSP m_opaque_sp;
};

} // namespace lldb

#endif // LLDB_SBInstructionList_h_
