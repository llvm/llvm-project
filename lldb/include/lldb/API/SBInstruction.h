//===-- SBInstruction.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBINSTRUCTION_H
#define LLDB_API_SBINSTRUCTION_H

#include "lldb/API/SBData.h"
#include "lldb/API/SBDefines.h"

#include <cstdio>

// There's a lot to be fixed here, but need to wait for underlying insn
// implementation to be revised & settle down first.

class InstructionImpl;

namespace lldb {

class LLDB_API SBInstruction {
public:
  SBInstruction();

  SBInstruction(const SBInstruction &rhs);

  const SBInstruction &operator=(const SBInstruction &rhs);

  ~SBInstruction();

  explicit operator bool() const;

  bool IsValid();

  SBAddress GetAddress();

  const char *GetMnemonic(lldb::SBTarget target);

  const char *GetOperands(lldb::SBTarget target);

  const char *GetComment(lldb::SBTarget target);

  lldb::InstructionControlFlowKind GetControlFlowKind(lldb::SBTarget target);

  lldb::SBData GetData(lldb::SBTarget target);

  size_t GetByteSize();

  bool DoesBranch();

  bool HasDelaySlot();

  bool CanSetBreakpoint();

#ifndef SWIG
  void Print(FILE *out);
#endif

  void Print(SBFile out);

  void Print(FileSP BORROWED);

  bool GetDescription(lldb::SBStream &description);

  bool EmulateWithFrame(lldb::SBFrame &frame, uint32_t evaluate_options);

  bool DumpEmulation(const char *triple); // triple is to specify the
                                          // architecture, e.g. 'armv6' or
                                          // 'armv7-apple-ios'

  bool TestEmulation(lldb::SBStream &output_stream, const char *test_file);

  /// Get variable annotations for this instruction as structured data.
  /// Returns an array of dictionaries, each containing:
  /// - "variable_name": string name of the variable
  /// - "location_description": string description of where variable is stored
  ///   ("RDI", "R15", "undef", etc.)
  /// - "start_address": unsigned integer address where this annotation becomes
  ///   valid
  /// - "end_address": unsigned integer address where this annotation becomes
  ///   invalid
  /// - "register_kind": unsigned integer indicating the register numbering
  /// scheme
  /// - "decl_file": string path to the file where variable is declared
  /// - "decl_line": unsigned integer line number where variable is declared
  /// - "type_name": string type name of the variable
  lldb::SBStructuredData GetVariableAnnotations();

protected:
  friend class SBInstructionList;

  SBInstruction(const lldb::DisassemblerSP &disasm_sp,
                const lldb::InstructionSP &inst_sp);

  void SetOpaque(const lldb::DisassemblerSP &disasm_sp,
                 const lldb::InstructionSP &inst_sp);

  lldb::InstructionSP GetOpaque();

private:
  std::shared_ptr<InstructionImpl> m_opaque_sp;
};

} // namespace lldb

#endif // LLDB_API_SBINSTRUCTION_H
