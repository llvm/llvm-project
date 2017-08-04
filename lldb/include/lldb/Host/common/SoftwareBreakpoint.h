//===-- SoftwareBreakpoint.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SoftwareBreakpoint_h_
#define liblldb_SoftwareBreakpoint_h_

#include "NativeBreakpoint.h"
#include "lldb/lldb-private-forward.h"

namespace lldb_private {
class SoftwareBreakpoint : public NativeBreakpoint {
  friend class NativeBreakpointList;

public:
  static Error CreateSoftwareBreakpoint(NativeProcessProtocol &process,
                                        lldb::addr_t addr, size_t size_hint,
                                        NativeBreakpointSP &breakpoint_spn);

  SoftwareBreakpoint(NativeProcessProtocol &process, lldb::addr_t addr,
                     const uint8_t *saved_opcodes, const uint8_t *trap_opcodes,
                     size_t opcode_size);

protected:
  Error DoEnable() override;

  Error DoDisable() override;

  bool IsSoftwareBreakpoint() const override;

private:
  /// Max number of bytes that a software trap opcode sequence can occupy.
  static const size_t MAX_TRAP_OPCODE_SIZE = 8;

  NativeProcessProtocol &m_process;
  uint8_t m_saved_opcodes[MAX_TRAP_OPCODE_SIZE];
  uint8_t m_trap_opcodes[MAX_TRAP_OPCODE_SIZE];
  const size_t m_opcode_size;

  static Error EnableSoftwareBreakpoint(NativeProcessProtocol &process,
                                        lldb::addr_t addr,
                                        size_t bp_opcode_size,
                                        const uint8_t *bp_opcode_bytes,
                                        uint8_t *saved_opcode_bytes);
};
}

#endif // #ifndef liblldb_SoftwareBreakpoint_h_
