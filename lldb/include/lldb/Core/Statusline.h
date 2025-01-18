//===-- Statusline.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_STATUSLINE_H
#define LLDB_CORE_STATUSLINE_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include <csignal>
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

  /// Redraw the statusline. If update is false, this will redraw the last
  /// string.
  void Redraw(bool update = true);

  /// Inform the statusline that the terminal dimensions have changed.
  void TerminalSizeChanged();

private:
  /// Draw the statusline with the given text.
  void Draw(std::string msg);

  /// Update terminal dimensions.
  void UpdateTerminalProperties();

  /// Set the scroll window to the given height.
  void SetScrollWindow(uint64_t height);

  /// Clear the statusline (without redrawing the background).
  void Reset();

  Debugger &m_debugger;
  std::string m_last_str;

  volatile std::sig_atomic_t m_terminal_size_has_changed = 1;
  uint64_t m_terminal_width = 0;
  uint64_t m_terminal_height = 0;
  uint64_t m_scroll_height = 0;
};
} // namespace lldb_private
#endif // LLDB_CORE_STATUSLINE_H
