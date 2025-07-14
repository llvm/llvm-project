//===-- Statusline.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Statusline.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/FormatEntity.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Locale.h"

#define ESCAPE "\x1b"
#define ANSI_NORMAL ESCAPE "[0m"
#define ANSI_SAVE_CURSOR ESCAPE "7"
#define ANSI_RESTORE_CURSOR ESCAPE "8"
#define ANSI_CLEAR_BELOW ESCAPE "[J"
#define ANSI_CLEAR_SCREEN ESCAPE "[2J"
#define ANSI_SET_SCROLL_ROWS ESCAPE "[1;%ur"
#define ANSI_TO_START_OF_ROW ESCAPE "[%u;1f"
#define ANSI_REVERSE_VIDEO ESCAPE "[7m"
#define ANSI_UP_ROWS ESCAPE "[%dA"

using namespace lldb;
using namespace lldb_private;

Statusline::Statusline(Debugger &debugger)
    : m_debugger(debugger), m_terminal_width(m_debugger.GetTerminalWidth()),
      m_terminal_height(m_debugger.GetTerminalHeight()) {
  Enable();
}

Statusline::~Statusline() { Disable(); }

void Statusline::TerminalSizeChanged() {
  m_terminal_width = m_debugger.GetTerminalWidth();
  m_terminal_height = m_debugger.GetTerminalHeight();

  UpdateScrollWindow(ResizeStatusline);

  // Draw the old statusline.
  Redraw(/*update=*/false);
}

void Statusline::Enable() {
  // Reduce the scroll window to make space for the status bar below.
  UpdateScrollWindow(EnableStatusline);

  // Draw the statusline.
  Redraw(/*update=*/true);
}

void Statusline::Disable() {
  // Extend the scroll window to cover the status bar.
  UpdateScrollWindow(DisableStatusline);
}

void Statusline::Draw(std::string str) {
  lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP();
  if (!stream_sp)
    return;

  m_last_str = str;

  str = ansi::TrimAndPad(str, m_terminal_width);

  LockedStreamFile locked_stream = stream_sp->Lock();
  locked_stream << ANSI_SAVE_CURSOR;
  locked_stream.Printf(ANSI_TO_START_OF_ROW,
                       static_cast<unsigned>(m_terminal_height));

  // Use "reverse video" to make sure the statusline has a background. Only do
  // this when colors are disabled, and rely on the statusline format otherwise.
  if (!m_debugger.GetUseColor())
    locked_stream << ANSI_REVERSE_VIDEO;

  locked_stream << str;
  locked_stream << ANSI_NORMAL;
  locked_stream << ANSI_RESTORE_CURSOR;
}

void Statusline::UpdateScrollWindow(ScrollWindowMode mode) {
  assert(m_terminal_width != 0 && m_terminal_height != 0);

  lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP();
  if (!stream_sp)
    return;

  const unsigned reduced_scroll_window = m_terminal_height - 1;
  LockedStreamFile locked_stream = stream_sp->Lock();

  switch (mode) {
  case EnableStatusline:
    // Move everything on the screen up.
    locked_stream << '\n';
    locked_stream.Printf(ANSI_UP_ROWS, 1);
    // Reduce the scroll window.
    locked_stream << ANSI_SAVE_CURSOR;
    locked_stream.Printf(ANSI_SET_SCROLL_ROWS, reduced_scroll_window);
    locked_stream << ANSI_RESTORE_CURSOR;
    break;
  case DisableStatusline:
    // Reset the scroll window.
    locked_stream << ANSI_SAVE_CURSOR;
    locked_stream.Printf(ANSI_SET_SCROLL_ROWS, 0);
    locked_stream << ANSI_RESTORE_CURSOR;
    // Clear the screen below to hide the old statusline.
    locked_stream << ANSI_CLEAR_BELOW;
    break;
  case ResizeStatusline:
    // Clear the screen and update the scroll window.
    // FIXME: Find a better solution (#146919).
    locked_stream << ANSI_CLEAR_SCREEN;
    locked_stream.Printf(ANSI_SET_SCROLL_ROWS, reduced_scroll_window);
    break;
  }

  m_debugger.RefreshIOHandler();
}

void Statusline::Redraw(bool update) {
  if (!update) {
    Draw(m_last_str);
    return;
  }

  ExecutionContext exe_ctx = m_debugger.GetSelectedExecutionContext();

  // For colors and progress events, the format entity needs access to the
  // debugger, which requires a target in the execution context.
  if (!exe_ctx.HasTargetScope())
    exe_ctx.SetTargetPtr(&m_debugger.GetSelectedOrDummyTarget());

  SymbolContext symbol_ctx;
  if (ProcessSP process_sp = exe_ctx.GetProcessSP()) {
    // Check if the process is stopped, and if it is, make sure it remains
    // stopped until we've computed the symbol context.
    Process::StopLocker stop_locker;
    if (stop_locker.TryLock(&process_sp->GetRunLock())) {
      if (auto frame_sp = exe_ctx.GetFrameSP())
        symbol_ctx = frame_sp->GetSymbolContext(eSymbolContextEverything);
    }
  }

  StreamString stream;
  FormatEntity::Entry format = m_debugger.GetStatuslineFormat();
  FormatEntity::Format(format, stream, &symbol_ctx, &exe_ctx, nullptr, nullptr,
                       false, false);

  Draw(stream.GetString().str());
}
