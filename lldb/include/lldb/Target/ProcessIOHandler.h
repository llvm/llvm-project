//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_PROCESSIOHANDLER_H
#define LLDB_TARGET_PROCESSIOHANDLER_H

#include "lldb/Core/IOHandler.h"
#include "lldb/Host/File.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Target/Process.h"

namespace lldb_private {

/// Forwards lldb's STDIN to the inferior's pty (or anything writable) and
/// supports asynchronous interrupt via an internal pipe. Used on POSIX hosts
/// and as a no-op stub on Windows.
class IOHandlerProcessSTDIO : public IOHandler {
public:
  IOHandlerProcessSTDIO(Process *process, int write_fd);

  ~IOHandlerProcessSTDIO() override = default;

  void SetIsRunning(bool running);

  void Run() override;

  void Cancel() override;

  bool Interrupt() override;

  void GotEOF() override {}

protected:
  Process *m_process;
  /// Read from this file (usually actual STDIN for LLDB)
  NativeFile m_read_file;
  /// Write to this file (usually the primary pty for getting io to debuggee)
  NativeFile m_write_file;
  Pipe m_pipe;
  std::mutex m_mutex;
  bool m_is_running = false;
};

#ifdef _WIN32

using HANDLE = void *;

/// Forwards lldb's STDIN to the inferior on Windows hosts. Reads from the
/// console (handling the line-buffering quirks of the Windows console) and
/// writes the bytes into the process via Process::PutSTDIN.
class IOHandlerProcessSTDIOWindows : public IOHandler {
public:
  IOHandlerProcessSTDIOWindows(Process *process);

  ~IOHandlerProcessSTDIOWindows() override;

  void SetIsRunning(bool running);

  /// Peek the console for input. If it has any, drain the pipe until text
  /// input is found or the pipe is empty.
  ///
  /// \param hStdin
  ///     The handle to the standard input's pipe.
  ///
  /// \return
  ///     true if the pipe has text input.
  llvm::Expected<bool> ConsoleHasTextInput(const HANDLE hStdin);

  void Run() override;

  void Cancel() override;

  bool Interrupt() override;

  void GotEOF() override {}

private:
  enum ControlOp : char {
    eControlOpQuit = 'q',
    eControlOpInterrupt = 'i',
    eControlOpNone = 0,
  };

  Process *m_process;
  /// Read from this file (usually actual STDIN for LLDB)
  NativeFile m_read_file;
  HANDLE m_interrupt_event =
      reinterpret_cast<HANDLE>(static_cast<intptr_t>(-1));
  std::atomic<ControlOp> m_pending_op{eControlOpNone};
  std::mutex m_mutex;
  bool m_is_running = false;
};

#endif // _WIN32

} // namespace lldb_private

#endif // LLDB_TARGET_PROCESSIOHANDLER_H
