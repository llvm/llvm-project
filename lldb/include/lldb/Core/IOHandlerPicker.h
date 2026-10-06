//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_IOHANDLERPICKER_H
#define LLDB_CORE_IOHANDLERPICKER_H

#include "lldb/Core/IOHandler.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Host/Terminal.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include <atomic>
#include <optional>
#include <string>

namespace lldb_private {

using PickerColumnName = std::string;

/// A row of the picker's table, with one value per column.
struct PickerItem {
  llvm::SmallVector<std::string> values;
  bool initially_selected = false;
};

enum class PickerMode { SingleSelect, MultiSelect };

struct PickerResult {
  bool was_canceled = false;
  /// Indices into the items given to the picker, in increasing order.
  llvm::SmallVector<size_t> selected_indices;
};

/// An interactive picker drawn as a dialog over the screen. When it closes, it
/// redraws what it covered from Debugger::GetTerminalHistory(). Reports a
/// canceled result if the input is not an interactive terminal.
class IOHandlerPicker : public IOHandler {
public:
  IOHandlerPicker(Debugger &debugger, llvm::StringRef header,
                  llvm::SmallVector<PickerColumnName> columns,
                  llvm::SmallVector<PickerItem, 0> items, PickerMode mode);

  ~IOHandlerPicker() override;

  void Run() override;
  void Cancel() override;
  bool Interrupt() override;
  void GotEOF() override;
  void TerminalSizeChanged() override;
  void Refresh() override;
  void PrintAsync(const char *s, size_t len, bool is_stdout) override;

  const PickerResult &GetResult() const { return m_result; }

private:
  enum class Key {
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Space,
    Enter,
    Sort,
    /// A byte arrived on m_wakeup_pipe.
    Wakeup,
    Escape,
    Unknown,
  };

  struct SortOrder {
    size_t column;
    bool ascending;
  };

  enum class Input { Ready, Wakeup, Error };

  bool EnterRawMode();
  void RestoreTerminalMode();

  Key ReadKey();
  Input WaitForInput();
  std::optional<char> ReadByte();
  Key ReadEscapeSequence();
  static Key DecodeByte(char c);
  static Key DecodeEscapeSequence(llvm::StringRef params, char final_byte);
  void WakeUp();

  void Render();
  /// Redraw the screen without the dialog.
  void RestoreScreen();
  void ScrollToCursor();
  void CycleSort();
  void UpdateVisibleIndices();
  void SortVisibleIndices(SortOrder order);
  void ComputeColumnWidths();
  void UpdateLayout();
  llvm::StringRef GetHelpText() const;
  size_t GetPrefixWidth() const;
  /// Width of the dialog without its padding.
  size_t GetInnerWidth() const;
  size_t GetViewportHeight() const { return m_viewport_height; }
  size_t GetDialogHeight() const;

  std::string m_header;
  llvm::SmallVector<PickerColumnName> m_columns;
  llvm::SmallVector<PickerItem, 0> m_items;
  PickerMode m_mode;
  PickerResult m_result;

  /// Index into m_visible_indices of the row under the cursor.
  size_t m_cursor_index = 0;
  /// Index into m_visible_indices of the first row on screen.
  size_t m_viewport_start = 0;
  /// Bit i is set if m_items[i] is checked.
  llvm::BitVector m_checked;

  /// Indices into m_items of the rows to show, in display order.
  llvm::SmallVector<size_t, 0> m_visible_indices;
  /// Sort column and direction, or std::nullopt for the original order.
  std::optional<SortOrder> m_sort_order;

  llvm::SmallVector<size_t> m_column_widths;

  /// Cached with the layout below so a frame never mixes two terminal sizes.
  size_t m_terminal_width;
  size_t m_terminal_height;

  /// Number of lines above the statusline.
  size_t m_screen_height = 0;
  /// Position and width of the dialog, 0-based.
  size_t m_box_top = 0;
  size_t m_box_left = 0;
  size_t m_box_width = 0;
  /// Number of item rows in the dialog.
  size_t m_viewport_height = 0;

  TerminalState m_saved_terminal_state;

  /// Wakes ReadKey() up for resizes and cancels from other threads, which can
  /// come before or after Run().
  Pipe m_wakeup_pipe;

  /// Lets Cancel() run on other threads, since only Run() writes m_result.
  std::atomic<bool> m_cancel_requested = false;
};

} // namespace lldb_private

#endif // LLDB_CORE_IOHANDLERPICKER_H
