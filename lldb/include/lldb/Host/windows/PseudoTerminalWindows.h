//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_Windows_PseudoTerminalWindows_H_
#define liblldb_Host_Windows_PseudoTerminalWindows_H_

#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/windows/windows.h"

namespace lldb_private {

class PseudoTerminalWindows : public PseudoTerminal {

public:
  void Close() override;

  HPCON GetPseudoTerminalHandle() override { return m_conpty_handle; };

  HANDLE GetPrimaryHandle() const override { return m_conpty_output; };

  HANDLE GetSecondaryHandle() const override { return m_conpty_input; };

  std::string GetSecondaryName() const override { return ""; };

  llvm::Error OpenFirstAvailablePrimary(int oflag) override;

protected:
  HANDLE m_conpty_handle = INVALID_HANDLE_VALUE;
  HANDLE m_conpty_output = INVALID_HANDLE_VALUE;
  HANDLE m_conpty_input = INVALID_HANDLE_VALUE;
};
}; // namespace lldb_private

#endif // liblldb_Host_Windows_PseudoTerminalWindows_H_
