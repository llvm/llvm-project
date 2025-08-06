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

  /// Redraw the statusline. If update is false, this will redraw the last
  /// string.
  void Redraw(bool update = true);

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
  std::string m_last_str;
  uint64_t m_terminal_width = 0;
  uint64_t m_terminal_height = 0;
};
} // namespace lldb_private
#endif // LLDB_CORE_STATUSLINE_H
