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

  // FIXME: This code gets called from a signal handler. It's probably not safe
  // to redraw the statusline, even without recomputing it?
  Redraw(/*update=*/false);
}

bool Statusline::IsSupported() const {
  File &file = m_debugger.GetOutputFile();
  return file.GetIsInteractive() && file.GetIsTerminalWithColors();
}

void Statusline::Enable() {
  if (!IsSupported())
    return;

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

void Statusline::Draw(llvm::StringRef str) {
  UpdateTerminalProperties();

  m_last_str = str;

  const size_t ellipsis = 3;
  const size_t column_width = ColumnWidth(str);

  if (column_width + ellipsis >= m_terminal_width)
    str = str.substr(0, m_terminal_width - ellipsis);

  StreamFile &out = m_debugger.GetOutputStream();
  out << ANSI_SAVE_CURSOR;
  out.Printf(ANSI_TO_START_OF_ROW, static_cast<unsigned>(m_terminal_height));
  out << ANSI_CLEAR_LINE;
  out << str;
  if (m_terminal_width > column_width)
    out << std::string(m_terminal_width - column_width, ' ');
  out << ansi::FormatAnsiTerminalCodes(k_ansi_suffix);
  out << ANSI_RESTORE_CURSOR;
}

void Statusline::Reset() {
  StreamFile &out = m_debugger.GetOutputStream();
  out << ANSI_SAVE_CURSOR;
  out.Printf(ANSI_TO_START_OF_ROW, static_cast<unsigned>(m_terminal_height));
  out << ANSI_CLEAR_LINE;
  out << ANSI_RESTORE_CURSOR;
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
  StreamFile &out = m_debugger.GetOutputStream();
  out << '\n';
  out << ANSI_SAVE_CURSOR;
  out.Printf(ANSI_SET_SCROLL_ROWS, static_cast<unsigned>(height));
  out << ANSI_RESTORE_CURSOR;
  out.Printf(ANSI_UP_ROWS, 1);
  out << ANSI_CLEAR_BELOW;
  out.Flush();

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

  Draw(stream.GetString());
}
