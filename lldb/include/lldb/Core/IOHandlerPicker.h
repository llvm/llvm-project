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
#include <functional>
#include <optional>
#include <string>

namespace lldb_private {

using PickerColumnName = std::string;

/// A row of the picker's table, with one value per column.
struct PickerItem {
  llvm::SmallVector<std::string> values;
  /// Checked in multi-select mode. In single-select mode, the cursor starts on
  /// the first such item.
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

  /// Returns the text to show under the list while the cursor is on the item
  /// at the given index, like the source code around a stack frame.
  using PreviewCallback = std::function<std::string(size_t index)>;
  void SetPreviewCallback(PreviewCallback callback) {
    m_preview_callback = std::move(callback);
  }

  /// Called with the result once the picker closes, on the thread that ran
  /// it. Lets the caller act on the result after RunIOHandlerAsync().
  using DoneCallback = std::function<void(const PickerResult &result)>;
  void SetDoneCallback(DoneCallback callback) {
    m_done_callback = std::move(callback);
  }

  /// Show a single-select picker and return the index of the chosen item, or
  /// std::nullopt if the user canceled or the input isn't a terminal.
  static std::optional<size_t>
  PickOne(Debugger &debugger, llvm::StringRef header,
          llvm::SmallVector<PickerColumnName> columns,
          llvm::SmallVector<PickerItem, 0> items,
          PreviewCallback preview_callback = {});

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
  void UpdatePreview();

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

  /// Indices into m_items of the rows that match m_query, in display order.
  llvm::SmallVector<size_t, 0> m_visible_indices;
  /// Sort column and direction, or std::nullopt for the original order.
  std::optional<SortOrder> m_sort_order;

  std::string m_query;
  bool m_searching = false;

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
  /// Number of preview lines in the dialog, or 0 without a preview.
  size_t m_preview_height = 0;

  PreviewCallback m_preview_callback;
  DoneCallback m_done_callback;
  /// Index into m_items of the item m_preview_lines is for.
  std::optional<size_t> m_preview_item;
  llvm::SmallVector<std::string> m_preview_lines;

  TerminalState m_saved_terminal_state;

  /// Wakes ReadKey() up for resizes and cancels from other threads, which can
  /// come before or after Run().
  Pipe m_wakeup_pipe;

  /// Lets Cancel() run on other threads, since only Run() writes m_result.
  std::atomic<bool> m_cancel_requested = false;
};

} // namespace lldb_private

#endif // LLDB_CORE_IOHANDLERPICKER_H
