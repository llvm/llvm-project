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
#include <mutex>
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

  /// Creates and opens a new ConPTY instance with a default console size of
  /// 80x25. Also sets up the associated STDIN/STDOUT pipes and drains any
  /// initialization sequences emitted by Windows.
  ///
  /// \return
  ///     An llvm::Error if the ConPTY could not be created, or if ConPTY is
  ///     not available on this version of Windows, llvm::Error::success()
  ///     otherwise.
  llvm::Error OpenPseudoConsole();

  /// Closes the ConPTY and invalidates its handle, without closing the STDIN
  /// and STDOUT pipes. Closing the ConPTY signals EOF to any process currently
  /// attached to it.
  void Close();

  /// Closes the STDIN and STDOUT pipe handles and invalidates them
  void ClosePipes();

  /// Returns whether the ConPTY and its pipes are currently open and valid.
  bool IsConnected() const;

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

  /// Drains initialization sequences from the ConPTY output pipe.
  ///
  /// When a process first attaches to a ConPTY, Windows emits VT100/ANSI escape
  /// sequences (ESC[2J for clear screen, ESC[H for cursor home and more) as
  /// part of the PseudoConsole initialization. To prevent these sequences from
  /// appearing in the debugger output (and flushing lldb's shell for instance)
  /// we launch a short-lived dummy process that triggers the initialization,
  /// then drain all output before launching the actual debuggee.
  llvm::Error DrainInitSequences();

  /// Returns a reference to the mutex used to synchronize access to the
  /// ConPTY state.
  std::mutex &GetMutex() { return m_mutex; };

  /// Returns a reference to the condition variable used to signal state changes
  /// to threads waiting on the ConPTY (e.g. waiting for output or shutdown).
  std::condition_variable &GetCV() { return m_cv; };

  /// Returns whether the ConPTY is in the process of shutting down.
  ///
  /// \return
  ///     A reference to the atomic bool that is set to true when the ConPTY
  ///     is stopping. Callers should check this in their read/write loops to
  ///     exit gracefully.
  const std::atomic<bool> &IsStopping() const { return m_stopping; };

  /// Sets the stopping flag to \p value, signalling to threads waiting on the
  /// ConPTY that they should stop.
  void SetStopping(bool value) { m_stopping = value; };

protected:
  HANDLE m_conpty_handle = ((HANDLE)(long long)-1);
  HANDLE m_conpty_output = ((HANDLE)(long long)-1);
  HANDLE m_conpty_input = ((HANDLE)(long long)-1);
  std::mutex m_mutex{};
  std::condition_variable m_cv{};
  std::atomic<bool> m_stopping = false;
};
} // namespace lldb_private

#endif // LIBLLDB_HOST_WINDOWS_PSEUDOCONSOLE_H_
