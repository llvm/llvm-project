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
      m_terminal_height(m_debugger.GetTerminalHeight()) {}

Statusline::~Statusline() { Disable(); }

void Statusline::TerminalSizeChanged() {
  m_terminal_width = m_debugger.GetTerminalWidth();
  m_terminal_height = m_debugger.GetTerminalHeight();

  UpdateScrollWindow(ResizeStatusline);

  // Redraw the old statusline.
  Redraw(std::nullopt);
}

void Statusline::Enable(std::optional<ExecutionContextRef> exe_ctx_ref) {
  // Reduce the scroll window to make space for the status bar below.
  UpdateScrollWindow(EnableStatusline);

  // Draw the statusline.
  Redraw(exe_ctx_ref);
}

void Statusline::Disable() {
  // Extend the scroll window to cover the status bar.
  UpdateScrollWindow(DisableStatusline);
}

void Statusline::Draw(std::string str) {
  lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP();
  if (!stream_sp)
    return;

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

void Statusline::Redraw(std::optional<ExecutionContextRef> exe_ctx_ref) {
  // Update the cached execution context.
  if (exe_ctx_ref)
    m_exe_ctx_ref = *exe_ctx_ref;

  // Lock the execution context.
  ExecutionContext exe_ctx =
      m_exe_ctx_ref.Lock(/*thread_and_frame_only_if_stopped=*/false);

  // Compute the symbol context if we're stopped.
  SymbolContext sym_ctx;
  llvm::Expected<StoppedExecutionContext> stopped_exe_ctx =
      GetStoppedExecutionContext(&m_exe_ctx_ref);
  if (stopped_exe_ctx) {
    // The StoppedExecutionContext only ensures that we hold the run lock.
    // The process could be in an exited or unloaded state and have no frame.
    if (auto frame_sp = stopped_exe_ctx->GetFrameSP())
      sym_ctx = frame_sp->GetSymbolContext(eSymbolContextEverything);
  } else {
    // We can draw the statusline without being stopped.
    llvm::consumeError(stopped_exe_ctx.takeError());
  }

  StreamString stream;
  FormatEntity::Entry format = m_debugger.GetStatuslineFormat();
  FormatEntity::Format(format, stream, &sym_ctx, &exe_ctx, nullptr, nullptr,
                       false, false);

  Draw(stream.GetString().str());
}
