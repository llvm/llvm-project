//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_Windows_PseudoConsole_H_
#define liblldb_Host_Windows_PseudoConsole_H_

#include "llvm/Support/Error.h"
#include <string>

#define PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE 0x20016

namespace lldb_private {

class PseudoConsole {

public:
  llvm::Error OpenPseudoConsole();

  /// Close the ConPTY, its read/write handles and invalidate them.
  void Close();

  /// The ConPTY HPCON handle accessor.
  ///
  /// This object retains ownership of the HPCON when this accessor is used.
  ///
  /// \return
  ///     The ConPTY HPCON handle, or INVALID_HANDLE_VALUE if it is currently
  ///     invalid.
  void *GetPseudoTerminalHandle() { return m_conpty_handle; };

  /// The STDOUT read HANDLE accessor.
  ///
  /// This object retains ownership of the HANDLE when this accessor is used.
  ///
  /// \return
  ///     The STDOUT read HANDLE, or INVALID_HANDLE_VALUE if it is currently
  ///     invalid.
  void *GetSTDOUTHandle() const { return m_conpty_output; };

  /// The STDIN write HANDLE accessor.
  ///
  /// This object retains ownership of the HANDLE when this accessor is used.
  ///
  /// \return
  ///     The STDIN write HANDLE, or INVALID_HANDLE_VALUE if it is currently
  ///     invalid.
  void *GetSTDINHandle() const { return m_conpty_input; };

protected:
  void *m_conpty_handle = ((void *)(long long)-1);
  void *m_conpty_output = ((void *)(long long)-1);
  void *m_conpty_input = ((void *)(long long)-1);
};
}; // namespace lldb_private

#endif // liblldb_Host_Windows_PseudoConsole_H_
