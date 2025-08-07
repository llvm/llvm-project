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

  /// Reduce the scroll window and draw the statusline.
  void Enable();

  /// Hide the statusline and extend the scroll window.
  void Disable();

  /// Redraw the statusline. If both exe_ctx and sym_ctx are NULL, this redraws
  /// the last string.
  void Redraw(const ExecutionContext *exe_ctx, const SymbolContext *sym_ctx);

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
  ExecutionContext m_exe_ctx;
  SymbolContext m_symbol_ctx;
  uint64_t m_terminal_width = 0;
  uint64_t m_terminal_height = 0;
};
} // namespace lldb_private
#endif // LLDB_CORE_STATUSLINE_H
