//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBLLDB_HOST_WINDOWS_PSEUDOCONSOLE_H_
#define LIBLLDB_HOST_WINDOWS_PSEUDOCONSOLE_H_

#include "llvm/Support/Error.h"
#include <string>

#define PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE 0x20016
typedef void *HANDLE;
typedef void *HPCON;

namespace lldb_private {

class PseudoConsole {

public:
  PseudoConsole() = default;
  ~PseudoConsole();

  PseudoConsole(const PseudoConsole &) = delete;
  PseudoConsole(PseudoConsole &&) = delete;
  PseudoConsole &operator=(const PseudoConsole &) = delete;
  PseudoConsole &operator=(PseudoConsole &&) = delete;

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
  HPCON GetPseudoTerminalHandle() { return m_conpty_handle; };

  /// The STDOUT read HANDLE accessor.
  ///
  /// This object retains ownership of the HANDLE when this accessor is used.
  ///
  /// \return
  ///     The STDOUT read HANDLE, or INVALID_HANDLE_VALUE if it is currently
  ///     invalid.
  HANDLE GetSTDOUTHandle() const { return m_conpty_output; };

  /// The STDIN write HANDLE accessor.
  ///
  /// This object retains ownership of the HANDLE when this accessor is used.
  ///
  /// \return
  ///     The STDIN write HANDLE, or INVALID_HANDLE_VALUE if it is currently
  ///     invalid.
  HANDLE GetSTDINHandle() const { return m_conpty_input; };

protected:
  HANDLE m_conpty_handle = ((HANDLE)(long long)-1);
  HANDLE m_conpty_output = ((HANDLE)(long long)-1);
  HANDLE m_conpty_input = ((HANDLE)(long long)-1);
};
} // namespace lldb_private

#endif // LIBLLDB_HOST_WINDOWS_PSEUDOCONSOLE_H_
