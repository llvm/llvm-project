//===-- SBInstructionList.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBInstructionList.h"
#include "SBReproducerPrivate.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

SBInstructionList::SBInstructionList() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBInstructionList);
}

SBInstructionList::SBInstructionList(const SBInstructionList &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBInstructionList, (const lldb::SBInstructionList &),
                          rhs);
}

const SBInstructionList &SBInstructionList::
operator=(const SBInstructionList &rhs) {
  LLDB_RECORD_METHOD(
      const lldb::SBInstructionList &,
      SBInstructionList, operator=,(const lldb::SBInstructionList &), rhs);

  if (this != &rhs)
    m_opaque_sp = rhs.m_opaque_sp;
  return *this;
}

SBInstructionList::~SBInstructionList() {}

bool SBInstructionList::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBInstructionList, IsValid);

  return m_opaque_sp.get() != NULL;
}

size_t SBInstructionList::GetSize() {
  LLDB_RECORD_METHOD_NO_ARGS(size_t, SBInstructionList, GetSize);

  if (m_opaque_sp)
    return m_opaque_sp->GetInstructionList().GetSize();
  return 0;
}

SBInstruction SBInstructionList::GetInstructionAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::SBInstruction, SBInstructionList,
                     GetInstructionAtIndex, (uint32_t), idx);

  SBInstruction inst;
  if (m_opaque_sp && idx < m_opaque_sp->GetInstructionList().GetSize())
    inst.SetOpaque(
        m_opaque_sp,
        m_opaque_sp->GetInstructionList().GetInstructionAtIndex(idx));
  return LLDB_RECORD_RESULT(inst);
}

size_t SBInstructionList::GetInstructionsCount(const SBAddress &start,
                                               const SBAddress &end,
                                               bool canSetBreakpoint) {
  LLDB_RECORD_METHOD(size_t, SBInstructionList, GetInstructionsCount,
                     (const lldb::SBAddress &, const lldb::SBAddress &, bool),
                     start, end, canSetBreakpoint);

  size_t num_instructions = GetSize();
  size_t i = 0;
  SBAddress addr;
  size_t lower_index = 0;
  size_t upper_index = 0;
  size_t instructions_to_skip = 0;
  for (i = 0; i < num_instructions; ++i) {
    addr = GetInstructionAtIndex(i).GetAddress();
    if (start == addr)
      lower_index = i;
    if (end == addr)
      upper_index = i;
  }
  if (canSetBreakpoint)
    for (i = lower_index; i <= upper_index; ++i) {
      SBInstruction insn = GetInstructionAtIndex(i);
      if (!insn.CanSetBreakpoint())
        ++instructions_to_skip;
    }
  return upper_index - lower_index - instructions_to_skip;
}

void SBInstructionList::Clear() {
  LLDB_RECORD_METHOD_NO_ARGS(void, SBInstructionList, Clear);

  m_opaque_sp.reset();
}

void SBInstructionList::AppendInstruction(SBInstruction insn) {
  LLDB_RECORD_METHOD(void, SBInstructionList, AppendInstruction,
                     (lldb::SBInstruction), insn);
}

void SBInstructionList::SetDisassembler(const lldb::DisassemblerSP &opaque_sp) {
  m_opaque_sp = opaque_sp;
}

void SBInstructionList::Print(FILE *out) {
  LLDB_RECORD_METHOD(void, SBInstructionList, Print, (FILE *), out);

  if (out == NULL)
    return;
}

bool SBInstructionList::GetDescription(lldb::SBStream &description) {
  LLDB_RECORD_METHOD(bool, SBInstructionList, GetDescription,
                     (lldb::SBStream &), description);

  if (m_opaque_sp) {
    size_t num_instructions = GetSize();
    if (num_instructions) {
      // Call the ref() to make sure a stream is created if one deesn't exist
      // already inside description...
      Stream &sref = description.ref();
      const uint32_t max_opcode_byte_size =
          m_opaque_sp->GetInstructionList().GetMaxOpcocdeByteSize();
      FormatEntity::Entry format;
      FormatEntity::Parse("${addr}: ", format);
      SymbolContext sc;
      SymbolContext prev_sc;
      for (size_t i = 0; i < num_instructions; ++i) {
        Instruction *inst =
            m_opaque_sp->GetInstructionList().GetInstructionAtIndex(i).get();
        if (inst == NULL)
          break;

        const Address &addr = inst->GetAddress();
        prev_sc = sc;
        ModuleSP module_sp(addr.GetModule());
        if (module_sp) {
          module_sp->ResolveSymbolContextForAddress(
              addr, eSymbolContextEverything, sc);
        }

        inst->Dump(&sref, max_opcode_byte_size, true, false, NULL, &sc,
                   &prev_sc, &format, 0);
        sref.EOL();
      }
      return true;
    }
  }
  return false;
}

bool SBInstructionList::DumpEmulationForAllInstructions(const char *triple) {
  LLDB_RECORD_METHOD(bool, SBInstructionList, DumpEmulationForAllInstructions,
                     (const char *), triple);

  if (m_opaque_sp) {
    size_t len = GetSize();
    for (size_t i = 0; i < len; ++i) {
      if (!GetInstructionAtIndex((uint32_t)i).DumpEmulation(triple))
        return false;
    }
  }
  return true;
}
