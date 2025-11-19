//===-- Statusline.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_STATUSLINE_H
#define LLDB_CORE_STATUSLINE_H

#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/lldb-forward.h"
#include <cstdint>
#include <string>

namespace lldb_private {
class Statusline {
public:
  Statusline(Debugger &debugger);
  ~Statusline();

  using Context = std::pair<ExecutionContextRef, SymbolContext>;

  /// Reduce the scroll window and draw the statusline.
  void Enable(std::optional<ExecutionContextRef> exe_ctx_ref);

  /// Hide the statusline and extend the scroll window.
  void Disable();

  /// Redraw the statusline.
  void Redraw(std::optional<ExecutionContextRef> exe_ctx_ref);

  /// Inform the statusline that the terminal dimensions have changed.
  void TerminalSizeChanged();

private:
  /// Draw the statusline with the given text.
  void Draw(std::string msg);

  enum ScrollWindowMode {
    EnableStatusline,
    DisableStatusline,
    ResizeStatusline,
  };

  /// Set the scroll window for the given mode.
  void UpdateScrollWindow(ScrollWindowMode mode);

  Debugger &m_debugger;

  /// Cached copy of the execution context that allows us to redraw the
  /// statusline.
  ExecutionContextRef m_exe_ctx_ref;

  uint64_t m_terminal_width = 0;
  uint64_t m_terminal_height = 0;
};
} // namespace lldb_private
#endif // LLDB_CORE_STATUSLINE_H
