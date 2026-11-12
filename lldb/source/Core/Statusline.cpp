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
#define ANSI_DISABLE_AUTO_WRAP ESCAPE "[?7l"
#define ANSI_ENABLE_AUTO_WRAP ESCAPE "[?7h"

using namespace lldb;
using namespace lldb_private;

Statusline::Statusline(Debugger &debugger)
    : m_debugger(debugger), m_terminal_width(m_debugger.GetTerminalWidth()),
      m_terminal_height(m_debugger.GetTerminalHeight()) {}

Statusline::~Statusline() { Disable(); }

void Statusline::TerminalSizeChanged() {
  // The dimensions the statusline was last drawn at, needed to clear it before
  // redrawing at the new size.
  const uint64_t prev_width = m_terminal_width;
  const uint64_t prev_height = m_terminal_height;

  m_terminal_width = m_debugger.GetTerminalWidth();
  m_terminal_height = m_debugger.GetTerminalHeight();

  UpdateScrollWindow(ResizeStatusline, prev_width, prev_height);

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
  // A statusline wider than the terminal (e.g. a stale width mid-resize) would
  // wrap onto the row above; with autowrap off it is clipped at the margin.
  locked_stream << ANSI_DISABLE_AUTO_WRAP;
  locked_stream.Printf(ANSI_TO_START_OF_ROW,
                       static_cast<unsigned>(m_terminal_height));

  // Use "reverse video" to make sure the statusline has a background. Only do
  // this when colors are disabled, and rely on the statusline format otherwise.
  if (!m_debugger.GetUseColor())
    locked_stream << ANSI_REVERSE_VIDEO;

  locked_stream << str;
  locked_stream << ANSI_NORMAL;
  locked_stream << ANSI_ENABLE_AUTO_WRAP;
  locked_stream << ANSI_RESTORE_CURSOR;
}

void Statusline::UpdateScrollWindow(ScrollWindowMode mode, uint64_t prev_width,
                                    uint64_t prev_height) {
  assert(m_terminal_width != 0 && m_terminal_height != 0);

  lldb::LockableStreamFileSP stream_sp = m_debugger.GetOutputStreamSP();
  if (!stream_sp)
    return;

  const unsigned reduced_scroll_rows = m_terminal_height - 1;
  { // Scope for locked_stream:
    LockedStreamFile locked_stream = stream_sp->Lock();

    switch (mode) {
    case EnableStatusline:
      // Move everything on the screen up.
      locked_stream << '\n';
      locked_stream.Printf(ANSI_UP_ROWS, 1);
      // Reduce the scroll window.
      locked_stream << ANSI_SAVE_CURSOR;
      locked_stream.Printf(ANSI_SET_SCROLL_ROWS, reduced_scroll_rows);
      locked_stream << ANSI_RESTORE_CURSOR;
      break;
    case DisableStatusline:
      // Reset the scroll window.
      locked_stream << ANSI_SAVE_CURSOR;
      locked_stream.Printf(ANSI_SET_SCROLL_ROWS,
                           static_cast<unsigned>(m_terminal_height));
      locked_stream << ANSI_RESTORE_CURSOR;
      // Clear the screen below to hide the old statusline.
      locked_stream << ANSI_CLEAR_BELOW;
      break;
    case ResizeStatusline: {
      // The old statusline is still on screen after a resize: a width shrink
      // reflows that full-width line into ceil(prev_width / width) rows at the
      // bottom, and growing taller strands it at its old row. Clear from the
      // topmost row it can occupy to the bottom (preserving the scrollback
      // above), then re-establish the scroll region. DECSTBM homes the cursor,
      // so save and restore it.
      const unsigned height = static_cast<unsigned>(m_terminal_height);
      unsigned reflow = 1;
      if (prev_width > m_terminal_width && m_terminal_width > 0)
        reflow = llvm::divideCeil(prev_width, m_terminal_width);
      if (reflow >= height)
        reflow = height - 1;
      unsigned first_row = height - reflow + 1;
      if (prev_height > 0 && prev_height < first_row)
        first_row = static_cast<unsigned>(prev_height);
      if (first_row < 1)
        first_row = 1;

      // A height shrink can leave the prompt on the row the statusline is about
      // to occupy, because the terminal reclaims the row below the cursor
      // instead of scrolling the cursor up. Scroll up one row to lift the
      // prompt clear, like EnableStatusline; the overlap is only ever the
      // single statusline row, and this is a no-op unless the cursor sits at
      // the bottom of the scroll region.
      if (prev_height > m_terminal_height) {
        locked_stream << '\n';
        locked_stream.Printf(ANSI_UP_ROWS, 1);
      }

      locked_stream << ANSI_SAVE_CURSOR;
      locked_stream.Printf(ANSI_TO_START_OF_ROW, first_row);
      locked_stream << ANSI_CLEAR_BELOW;
      locked_stream.Printf(ANSI_SET_SCROLL_ROWS, reduced_scroll_rows);
      locked_stream << ANSI_RESTORE_CURSOR;
      break;
    }
    }
  }
  m_debugger.RefreshIOHandler();
}

void Statusline::ClearExecutionContext() { m_exe_ctx_ref.ClearFrame(); }

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
  FormatEntity::Formatter(&sym_ctx, &exe_ctx, nullptr, false, false)
      .Format(format, stream);

  Draw(stream.GetString().str());
}
