//===-- ConnectionConPTY.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_CONNECTIONCONPTYWINDOWS_H
#define LLDB_HOST_WINDOWS_CONNECTIONCONPTYWINDOWS_H

#include "lldb/Host/windows/ConnectionGenericFileWindows.h"
#include "lldb/Host/windows/PseudoConsole.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/Connection.h"
#include "lldb/lldb-types.h"
#include <mutex>

namespace lldb_private {

/// A read only Connection implementation for the Windows ConPTY.
class ConnectionConPTY : public ConnectionGenericFile {
public:
  ConnectionConPTY(std::shared_ptr<PseudoConsole> pty);

  ~ConnectionConPTY();

  lldb::ConnectionStatus Connect(llvm::StringRef s, Status *error_ptr) override;

  lldb::ConnectionStatus Disconnect(Status *error_ptr) override;

  /// Read from the ConPTY's pipe.
  ///
  /// Before reading, check if the ConPTY is closing and wait for it to close
  /// before reading. This prevents race conditions when closing the ConPTY
  /// during a read. After reading, remove the ConPTY VT init sequence if
  /// present.
  size_t Read(void *dst, size_t dst_len, const Timeout<std::micro> &timeout,
              lldb::ConnectionStatus &status, Status *error_ptr) override;

  size_t Write(const void *src, size_t src_len, lldb::ConnectionStatus &status,
               Status *error_ptr) override;

protected:
  std::shared_ptr<PseudoConsole> m_pty;
  bool m_pty_vt_sequence_was_stripped = false;
};
} // namespace lldb_private

#endif
