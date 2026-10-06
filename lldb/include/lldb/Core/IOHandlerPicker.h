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

#include <optional>
#include <string>

namespace lldb_private {

using PickerColumn = std::string;

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

/// An interactive ANSI-based picker IOHandler.
///
/// Supports single-select and multi-select modes, column sorting, filtering
/// rows by a search query, vim-style navigation, and scrolling. Works in any
/// terminal that supports basic ANSI escape codes. Reports a canceled result
/// if the input is not an interactive terminal.
class IOHandlerPicker : public IOHandler {
public:
  IOHandlerPicker(Debugger &debugger, llvm::StringRef header,
                  llvm::SmallVector<PickerColumn> columns,
                  llvm::SmallVector<PickerItem, 0> items, PickerMode mode);

  ~IOHandlerPicker() override;

  void Run() override;
  void Cancel() override;
  bool Interrupt() override;
  void GotEOF() override;
  void TerminalSizeChanged() override;

  const PickerResult &GetResult() const { return m_result; }

private:
  enum class Key {
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Enter,
    Backspace,
    Escape,
    /// Ctrl+C or a read failure, which always cancel the picker.
    Abort,
    /// A byte arrived on m_wakeup_pipe.
    Wakeup,
    /// A printable character, stored in KeyPress::ch.
    Char,
    Unknown,
  };

  struct KeyPress {
    Key key;
    char ch = 0;
  };

  struct SortOrder {
    size_t column;
    bool ascending;
  };

  enum class Input { Ready, Wakeup, Error };

  /// Raw mode hands every key press to ReadKey() as soon as it's typed.
  bool EnterRawMode();
  void RestoreTerminalMode();

  KeyPress ReadKey();
  Input WaitForInput();
  std::optional<char> ReadByte();
  Key ReadEscapeSequence();
  static KeyPress DecodeByte(char c);
  static Key DecodeEscapeSequence(llvm::StringRef params, char final_byte);
  void WakeUp();

  void HandleKey(KeyPress press);
  void HandleSearchKey(KeyPress press);
  void Render();
  void ScrollToCursor();
  void CycleSort();
  void UpdateVisibleIndices();
  void SortVisibleIndices(SortOrder order);
  void ComputeColumnWidths();
  void UpdateLayout();
  size_t GetPrefixWidth() const;
  size_t GetViewportHeight() const;

  std::string m_header;
  llvm::SmallVector<PickerColumn> m_columns;
  llvm::SmallVector<PickerItem, 0> m_items;
  PickerMode m_mode;
  PickerResult m_result;

  /// Index into m_visible_indices of the row under the cursor.
  size_t m_cursor_index = 0;
  /// Index into m_visible_indices of the first row on screen.
  size_t m_viewport_start = 0;
  /// Bit i is set if m_items[i] is checked.
  llvm::BitVector m_checked;

  /// Indices into m_items of the rows that match m_query, in display order.
  llvm::SmallVector<size_t, 0> m_visible_indices;
  /// Sort column and direction, or std::nullopt for the original order.
  std::optional<SortOrder> m_sort_order;

  std::string m_query;
  bool m_searching = false;

  llvm::SmallVector<size_t> m_column_widths;

  /// Cached with m_column_widths so a frame never mixes two terminal sizes.
  size_t m_terminal_width;
  size_t m_terminal_height;

  TerminalState m_saved_terminal_state;

  /// Lets TerminalSizeChanged() and Cancel(), which can run on other threads,
  /// wake ReadKey() up from select(). Open for the picker's whole lifetime,
  /// since TerminalSizeChanged() can come before or after Run().
  Pipe m_wakeup_pipe;
};

} // namespace lldb_private

#endif // LLDB_CORE_IOHANDLERPICKER_H
