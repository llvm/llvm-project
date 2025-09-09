//===-- StackID.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_STACKID_H
#define LLDB_TARGET_STACKID_H

#include "lldb/Core/AddressRange.h"

namespace lldb_private {

class Process;

class StackID {
public:
  StackID() = default;

  explicit StackID(lldb::addr_t pc, lldb::addr_t cfa,
                   SymbolContextScope *symbol_scope, Process *process);

  ~StackID() = default;

  lldb::addr_t GetPC() const { return m_pc; }

  lldb::addr_t GetCallFrameAddress() const { return m_cfa; }

  SymbolContextScope *GetSymbolContextScope() const { return m_symbol_scope; }

  void SetSymbolContextScope(SymbolContextScope *symbol_scope) {
    m_symbol_scope = symbol_scope;
  }

  void Clear() {
    m_pc = LLDB_INVALID_ADDRESS;
    m_cfa = LLDB_INVALID_ADDRESS;
    m_symbol_scope = nullptr;
  }

  bool IsValid() const {
    return m_pc != LLDB_INVALID_ADDRESS || m_cfa != LLDB_INVALID_ADDRESS;
  }

  void Dump(Stream *s);

protected:
  friend class StackFrame;

  void SetPC(lldb::addr_t pc, Process *process);
  void SetCFA(lldb::addr_t cfa, Process *process);

  /// The pc value for the function/symbol for this frame. This will only get
  /// used if the symbol scope is nullptr (the code where we are stopped is not
  /// represented by any function or symbol in any shared library).
  lldb::addr_t m_pc = LLDB_INVALID_ADDRESS;

  /// The call frame address (stack pointer) value at the beginning of the
  /// function that uniquely identifies this frame (along with m_symbol_scope
  /// below)
  lldb::addr_t m_cfa = LLDB_INVALID_ADDRESS;

  /// If nullptr, there is no block or symbol for this frame. If not nullptr,
  /// this will either be the scope for the lexical block for the frame, or the
  /// scope for the symbol. Symbol context scopes are always be unique pointers
  /// since the are part of the Block and Symbol objects and can easily be used
  /// to tell if a stack ID is the same as another.
  SymbolContextScope *m_symbol_scope = nullptr;
};

bool operator==(const StackID &lhs, const StackID &rhs);
bool operator!=(const StackID &lhs, const StackID &rhs);

// frame_id_1 < frame_id_2 means "frame_id_1 is YOUNGER than frame_id_2"
bool operator<(const StackID &lhs, const StackID &rhs);

} // namespace lldb_private

#endif // LLDB_TARGET_STACKID_H
