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
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/Support/Locale.h"

#define ESCAPE "\x1b"
#define ANSI_NORMAL ESCAPE "[0m"
#define ANSI_SAVE_CURSOR ESCAPE "7"
#define ANSI_RESTORE_CURSOR ESCAPE "8"
#define ANSI_CLEAR_BELOW ESCAPE "[J"
#define ANSI_CLEAR_LINE "\r\x1B[2K"
#define ANSI_SET_SCROLL_ROWS ESCAPE "[0;%ur"
#define ANSI_TO_START_OF_ROW ESCAPE "[%u;0f"
#define ANSI_UP_ROWS ESCAPE "[%dA"
#define ANSI_DOWN_ROWS ESCAPE "[%dB"
#define ANSI_FORWARD_COLS ESCAPE "\033[%dC"
#define ANSI_BACKWARD_COLS ESCAPE "\033[%dD"

using namespace lldb;
using namespace lldb_private;

static size_t ColumnWidth(llvm::StringRef str) {
  std::string stripped = ansi::StripAnsiTerminalCodes(str);
  return llvm::sys::locale::columnWidth(stripped);
}

Statusline::Statusline(Debugger &debugger) : m_debugger(debugger) { Enable(); }

Statusline::~Statusline() { Disable(); }

void Statusline::TerminalSizeChanged() {
  m_terminal_size_has_changed = 1;

  // This definitely isn't signal safe, but the best we can do, until we
  // have proper signal-catching thread.
  Redraw(/*update=*/false);
}

void Statusline::Enable() {
  UpdateTerminalProperties();

  // Reduce the scroll window to make space for the status bar below.
  SetScrollWindow(m_terminal_height - 1);

  // Draw the statusline.
  Redraw();
}

void Statusline::Disable() {
  UpdateTerminalProperties();

  // Extend the scroll window to cover the status bar.
  SetScrollWindow(m_terminal_height);
}

void Statusline::Draw(std::string str) {
  static constexpr const size_t g_ellipsis = 3;

  UpdateTerminalProperties();

  m_last_str = str;

  size_t column_width = ColumnWidth(str);

  if (column_width + g_ellipsis >= m_terminal_width) {
    // FIXME: If there are hidden characters (e.g. UTF-8, ANSI escape
    // characters), this will strip the string more than necessary. Ideally we
    // want to strip until column_width == m_terminal_width.
    str = str.substr(0, m_terminal_width);
    str.replace(m_terminal_width - g_ellipsis, g_ellipsis, "...");
    column_width = ColumnWidth(str);
  }

  if (lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP()) {
    LockedStreamFile locked_stream = stream_sp->Lock();
    locked_stream << ANSI_SAVE_CURSOR;
    locked_stream.Printf(ANSI_TO_START_OF_ROW,
                         static_cast<unsigned>(m_terminal_height));
    locked_stream << ANSI_CLEAR_LINE;
    locked_stream << str;
    if (column_width < m_terminal_width)
      locked_stream << std::string(m_terminal_width - column_width, ' ');
    locked_stream << ANSI_NORMAL;
    locked_stream << ANSI_RESTORE_CURSOR;
  }
}

void Statusline::Reset() {
  if (lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP()) {
    LockedStreamFile locked_stream = stream_sp->Lock();
    locked_stream << ANSI_SAVE_CURSOR;
    locked_stream.Printf(ANSI_TO_START_OF_ROW,
                         static_cast<unsigned>(m_terminal_height));
    locked_stream << ANSI_CLEAR_LINE;
    locked_stream << ANSI_RESTORE_CURSOR;
  }
}

void Statusline::UpdateTerminalProperties() {
  if (m_terminal_size_has_changed == 0)
    return;

  // Clear the previous statusline using the previous dimensions.
  Reset();

  m_terminal_width = m_debugger.GetTerminalWidth();
  m_terminal_height = m_debugger.GetTerminalHeight();

  // Set the scroll window based on the new terminal height.
  SetScrollWindow(m_terminal_height - 1);

  // Clear the flag.
  m_terminal_size_has_changed = 0;
}

void Statusline::SetScrollWindow(uint64_t height) {
  if (lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP()) {
    LockedStreamFile locked_stream = stream_sp->Lock();
    locked_stream << '\n';
    locked_stream << ANSI_SAVE_CURSOR;
    locked_stream.Printf(ANSI_SET_SCROLL_ROWS, static_cast<unsigned>(height));
    locked_stream << ANSI_RESTORE_CURSOR;
    locked_stream.Printf(ANSI_UP_ROWS, 1);
    locked_stream << ANSI_CLEAR_BELOW;
  }

  m_scroll_height = height;
}

void Statusline::Redraw(bool update) {
  if (!update) {
    Draw(m_last_str);
    return;
  }

  StreamString stream;
  ExecutionContext exe_ctx =
      m_debugger.GetCommandInterpreter().GetExecutionContext();

  // For colors and progress events, the format entity needs access to the
  // debugger, which requires a target in the execution context.
  if (!exe_ctx.HasTargetScope())
    exe_ctx.SetTargetPtr(&m_debugger.GetSelectedOrDummyTarget());

  SymbolContext symbol_ctx;
  if (auto frame_sp = exe_ctx.GetFrameSP())
    symbol_ctx = frame_sp->GetSymbolContext(eSymbolContextEverything);

  if (auto *format = m_debugger.GetStatuslineFormat())
    FormatEntity::Format(*format, stream, &symbol_ctx, &exe_ctx, nullptr,
                         nullptr, false, false);

  Draw(std::string(stream.GetString()));
}
