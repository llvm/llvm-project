//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/IOHandlerPicker.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/SelectHelper.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Locale.h"

#if LLDB_ENABLE_TERMIOS
#include <unistd.h>
#endif

#include <algorithm>
#include <cassert>
#include <chrono>

using namespace lldb;
using namespace lldb_private;

static constexpr size_t g_column_gap = 2;

static constexpr llvm::StringLiteral g_checked_prefix = " [x] ";
static constexpr llvm::StringLiteral g_unchecked_prefix = " [ ] ";
static constexpr llvm::StringLiteral g_cursor_prefix = " > ";
static constexpr llvm::StringLiteral g_no_cursor_prefix = "   ";

static constexpr llvm::StringLiteral g_ascending_indicator = " ^";
static constexpr llvm::StringLiteral g_descending_indicator = " v";
static_assert(g_ascending_indicator.size() == g_descending_indicator.size(),
              "both sort indicators must take the same width");

/// Header, key help, column names, footer, and an empty last line so that
/// the footer's newline doesn't scroll the screen.
static constexpr size_t g_reserved_lines = 5;

static constexpr char g_ctrl_c = 0x03;
static constexpr char g_escape = 0x1b;
static constexpr char g_delete = 0x7f; // Backspace on most terminals.

static constexpr std::chrono::milliseconds g_escape_sequence_timeout(100);

/// Parameter (0x30-0x3f) or intermediate (0x20-0x2f) byte, see ECMA-48 5.4.
static bool IsCSIParameterOrIntermediateByte(char c) {
  return c >= 0x20 && c <= 0x3f;
}

/// Falls back to the size in bytes when columnWidth() can't measure \a str.
static size_t GetColumnWidth(llvm::StringRef str) {
  int width = llvm::sys::locale::columnWidth(str);
  return width < 0 ? str.size() : width;
}

IOHandlerPicker::IOHandlerPicker(Debugger &debugger, llvm::StringRef header,
                                 llvm::SmallVector<PickerColumn> columns,
                                 llvm::SmallVector<PickerItem, 0> items,
                                 PickerMode mode)
    : IOHandler(debugger, IOHandler::Type::Picker), m_header(header.str()),
      m_columns(std::move(columns)), m_items(std::move(items)), m_mode(mode),
      m_checked(m_items.size()), m_terminal_width(debugger.GetTerminalWidth()),
      m_terminal_height(debugger.GetTerminalHeight()) {
  assert(llvm::all_of(m_items,
                      [&](const PickerItem &item) {
                        return item.values.size() == m_columns.size();
                      }) &&
         "every picker item must have one value per column");

  for (auto [i, item] : llvm::enumerate(m_items))
    if (item.initially_selected)
      m_checked.set(i);

  UpdateVisibleIndices();

  // Without the pipe, resizes and cross-thread cancels wait for a key press.
  if (Status error = m_wakeup_pipe.CreateNew(); error.Fail())
    LLDB_LOG(GetLog(LLDBLog::Host), "IOHandlerPicker: {0}", error);
}

IOHandlerPicker::~IOHandlerPicker() = default;

size_t IOHandlerPicker::GetPrefixWidth() const {
  return m_mode == PickerMode::MultiSelect ? g_checked_prefix.size()
                                           : g_cursor_prefix.size();
}

void IOHandlerPicker::ComputeColumnWidths() {
  m_column_widths.clear();
  for (size_t i = 0; i < m_columns.size(); ++i) {
    size_t width = GetColumnWidth(m_columns[i]) + g_ascending_indicator.size();
    for (const PickerItem &item : m_items)
      width = std::max(width, GetColumnWidth(item.values[i]));
    m_column_widths.push_back(width);
  }

  size_t row_width = GetPrefixWidth();
  for (size_t width : m_column_widths)
    row_width += width + g_column_gap;
  if (m_column_widths.empty() || row_width >= m_terminal_width)
    return;

  // Spread the unused width evenly, the first columns taking the remainder.
  size_t num_cols = m_column_widths.size();
  size_t extra = m_terminal_width - row_width;
  for (size_t i = 0; i < num_cols; ++i) {
    m_column_widths[i] += extra / num_cols;
    if (i < extra % num_cols)
      m_column_widths[i] += 1;
  }
}

void IOHandlerPicker::UpdateLayout() {
  m_terminal_width = m_debugger.GetTerminalWidth();
  m_terminal_height = m_debugger.GetTerminalHeight();
  ComputeColumnWidths();
}

size_t IOHandlerPicker::GetViewportHeight() const {
  // Even on a tiny terminal, keep the cursor's row on screen.
  if (m_terminal_height <= g_reserved_lines)
    return 1;
  return m_terminal_height - g_reserved_lines;
}

bool IOHandlerPicker::EnterRawMode() {
#if LLDB_ENABLE_TERMIOS
  int fd = GetInputFD();
  if (fd < 0)
    return false;

  Terminal terminal(fd);
  m_saved_terminal_state.Save(terminal, /*save_process_group=*/false);
  if (auto err = terminal.SetRaw()) {
    llvm::consumeError(std::move(err));
    m_saved_terminal_state.Clear();
    return false;
  }
  return true;
#else
  return false;
#endif
}

void IOHandlerPicker::RestoreTerminalMode() {
  if (m_saved_terminal_state.IsValid()) {
    m_saved_terminal_state.Restore();
    m_saved_terminal_state.Clear();
  }
}

IOHandlerPicker::KeyPress IOHandlerPicker::ReadKey() {
  switch (WaitForInput()) {
  case Input::Ready:
    break;
  case Input::Wakeup:
    return {Key::Wakeup};
  case Input::Error:
    return {Key::Abort};
  }

  std::optional<char> c = ReadByte();
  if (!c)
    return {Key::Abort};
  if (*c == g_escape)
    return {ReadEscapeSequence()};
  return DecodeByte(*c);
}

IOHandlerPicker::Input IOHandlerPicker::WaitForInput() {
  bool has_wakeup_pipe = m_wakeup_pipe.CanRead();
  int wakeup_fd = m_wakeup_pipe.GetReadFileDescriptor();
  SelectHelper wait_for_input;
  wait_for_input.FDSetRead(GetInputFD());
  if (has_wakeup_pipe)
    wait_for_input.FDSetRead(wakeup_fd);
  if (wait_for_input.Select().Fail())
    return Input::Error;
  if (!has_wakeup_pipe || !wait_for_input.FDIsSetRead(wakeup_fd))
    return Input::Ready;

  // Drain the pipe so the next select() blocks.
  char buf[32];
  if (llvm::Expected<size_t> bytes = m_wakeup_pipe.Read(buf, sizeof(buf));
      !bytes)
    llvm::consumeError(bytes.takeError());
  return Input::Wakeup;
}

std::optional<char> IOHandlerPicker::ReadByte() {
#if LLDB_ENABLE_TERMIOS
  char c;
  if (::read(GetInputFD(), &c, 1) == 1)
    return c;
#endif
  return std::nullopt;
}

IOHandlerPicker::Key IOHandlerPicker::ReadEscapeSequence() {
  // A lone ESC is the ESC key. A sequence's bytes follow right away.
  SelectHelper wait_for_sequence;
  wait_for_sequence.SetTimeout(g_escape_sequence_timeout);
  wait_for_sequence.FDSetRead(GetInputFD());
  if (wait_for_sequence.Select().Fail())
    return Key::Escape;

  // CSI ("ESC [") or SS3 ("ESC O").
  std::optional<char> intro = ReadByte();
  if (intro != '[' && intro != 'O')
    return Key::Unknown;

  // Consume the whole sequence, even for keys we ignore.
  std::string params;
  std::optional<char> final_byte = ReadByte();
  while (intro == '[' && final_byte &&
         IsCSIParameterOrIntermediateByte(*final_byte)) {
    params += *final_byte;
    final_byte = ReadByte();
  }
  if (!final_byte)
    return Key::Abort;
  return DecodeEscapeSequence(params, *final_byte);
}

IOHandlerPicker::KeyPress IOHandlerPicker::DecodeByte(char c) {
  switch (c) {
  case '\r':
  case '\n':
    return {Key::Enter};
  case '\b':
  case g_delete:
    return {Key::Backspace};
  case g_ctrl_c: // Raw mode disables SIGINT, so Ctrl+C arrives as a byte.
    return {Key::Abort};
  default:
    if (llvm::isPrint(c))
      return {Key::Char, c};
    return {Key::Unknown};
  }
}

IOHandlerPicker::Key
IOHandlerPicker::DecodeEscapeSequence(llvm::StringRef params, char final_byte) {
  switch (final_byte) {
  case 'A':
    return Key::Up;
  case 'B':
    return Key::Down;
  case 'H':
    return Key::Home;
  case 'F':
    return Key::End;
  case '~':
    // "ESC [ <code>[;<modifiers>] ~". Home/End: 1/4 on vt220, 7/8 on rxvt.
    return llvm::StringSwitch<Key>(params.split(';').first)
        .Cases({"1", "7"}, Key::Home)
        .Cases({"4", "8"}, Key::End)
        .Case("5", Key::PageUp)
        .Case("6", Key::PageDown)
        .Default(Key::Unknown);
  default:
    return Key::Unknown;
  }
}

void IOHandlerPicker::ScrollToCursor() {
  size_t viewport = GetViewportHeight();

  if (m_cursor_index < m_viewport_start)
    m_viewport_start = m_cursor_index;
  else if (m_cursor_index >= m_viewport_start + viewport)
    m_viewport_start = m_cursor_index - viewport + 1;
}

void IOHandlerPicker::CycleSort() {
  // unsorted -> col0 asc -> col0 desc -> col1 asc -> ... -> unsorted
  if (m_columns.empty())
    return;
  if (!m_sort_order)
    m_sort_order = SortOrder{0, /*ascending=*/true};
  else if (m_sort_order->ascending)
    m_sort_order->ascending = false;
  else if (m_sort_order->column + 1 < m_columns.size())
    m_sort_order = SortOrder{m_sort_order->column + 1, /*ascending=*/true};
  else
    m_sort_order.reset();
  UpdateVisibleIndices();
}

void IOHandlerPicker::UpdateVisibleIndices() {
  // Start from the item order, which the stable sort keeps for ties.
  m_visible_indices.clear();
  for (auto [i, item] : llvm::enumerate(m_items)) {
    if (m_query.empty() ||
        llvm::any_of(item.values, [&](llvm::StringRef value) {
          return value.contains_insensitive(m_query);
        }))
      m_visible_indices.push_back(i);
  }
  if (m_sort_order)
    SortVisibleIndices(*m_sort_order);

  // Jump to the top so the new order or matches are visible.
  m_cursor_index = 0;
  m_viewport_start = 0;
}

void IOHandlerPicker::SortVisibleIndices(SortOrder order) {
  size_t col = order.column;
  // Compare as integers if every value in the column is one, so that "9"
  // sorts before "10".
  llvm::SmallVector<int64_t, 0> numbers;
  numbers.reserve(m_items.size());
  for (const PickerItem &item : m_items) {
    int64_t number;
    if (llvm::StringRef(item.values[col]).getAsInteger(10, number))
      break;
    numbers.push_back(number);
  }
  bool numeric = numbers.size() == m_items.size();
  auto less = [&](size_t a, size_t b) {
    if (numeric)
      return numbers[a] < numbers[b];
    return m_items[a].values[col] < m_items[b].values[col];
  };
  llvm::stable_sort(m_visible_indices, [&](size_t a, size_t b) {
    return order.ascending ? less(a, b) : less(b, a);
  });
}

void IOHandlerPicker::Render() {
  // Build the whole frame first so it reaches the terminal in one write.
  std::string frame = ANSI_CSI_RESET_CURSOR;

  // Fit every line to the terminal width so it overwrites the previous frame.
  auto write_line = [&](llvm::StringRef style, llvm::StringRef content) {
    frame += style;
    frame += ansi::TrimAndPad(content, m_terminal_width);
    frame += ANSI_ESCAPE1(ANSI_CTRL_NORMAL) "\r\n";
  };

  write_line(ANSI_ESCAPE1(ANSI_CTRL_BOLD), m_header);

  if (m_searching) {
    std::string prompt = "  /" + m_query + "_";
    prompt += ANSI_ESCAPE1(ANSI_CTRL_FAINT) "    Enter: apply  ESC: clear";
    write_line("", prompt);
  } else {
    llvm::StringRef help =
        m_mode == PickerMode::SingleSelect
            ? "  jk: navigate  Enter: select  s: sort  /: search  ESC: cancel"
            : "  jk: navigate  Space: toggle  Enter: confirm  s: sort  "
              "/: search  ESC: cancel";
    write_line(ANSI_ESCAPE1(ANSI_CTRL_FAINT), help);
  }

  std::string column_names(GetPrefixWidth(), ' ');
  for (size_t i = 0; i < m_columns.size(); ++i) {
    std::string name = m_columns[i];
    bool is_sort_col = m_sort_order && m_sort_order->column == i;
    if (is_sort_col) {
      name += m_sort_order->ascending ? g_ascending_indicator
                                      : g_descending_indicator;
      column_names += ANSI_ESCAPE1(ANSI_CTRL_NORMAL)
          ANSI_ESCAPE1(ANSI_CTRL_BOLD) ANSI_ESCAPE1(ANSI_CTRL_UNDERLINE);
    }
    column_names += name;
    if (is_sort_col)
      column_names +=
          ANSI_ESCAPE1(ANSI_CTRL_NORMAL) ANSI_ESCAPE1(ANSI_CTRL_FAINT);
    column_names.append(
        m_column_widths[i] - GetColumnWidth(name) + g_column_gap, ' ');
  }
  write_line(ANSI_ESCAPE1(ANSI_CTRL_FAINT), column_names);

  size_t end = std::min(m_viewport_start + GetViewportHeight(),
                        m_visible_indices.size());
  for (size_t i = m_viewport_start; i < end; ++i) {
    size_t item_idx = m_visible_indices[i];
    bool is_cursor = i == m_cursor_index;

    std::string row;
    if (m_mode == PickerMode::MultiSelect)
      row = (m_checked[item_idx] ? g_checked_prefix : g_unchecked_prefix).str();
    else
      row = (is_cursor ? g_cursor_prefix : g_no_cursor_prefix).str();

    for (auto [value, width] :
         llvm::zip_equal(m_items[item_idx].values, m_column_widths)) {
      row += value;
      row.append(width - GetColumnWidth(value) + g_column_gap, ' ');
    }
    write_line(is_cursor ? ANSI_ESCAPE1(ANSI_CTRL_IMAGE_NEGATIVE) : "", row);
  }

  std::string footer;
  if (m_mode == PickerMode::MultiSelect)
    footer =
        llvm::formatv("  {0}/{1} selected", m_checked.count(), m_items.size())
            .str();
  else if (!m_visible_indices.empty())
    footer =
        llvm::formatv("  {0}/{1}", m_cursor_index + 1, m_visible_indices.size())
            .str();
  if (m_visible_indices.empty())
    footer += "  no matches";
  else if (!m_query.empty() && !m_searching)
    footer += "  filter: " + m_query;
  if (size_t hidden = m_items.size() - m_visible_indices.size())
    footer += llvm::formatv("  ({0} hidden)", hidden).str();
  if (end < m_visible_indices.size())
    footer += "  [more below]";
  if (m_viewport_start > 0)
    footer += "  [more above]";
  write_line(ANSI_ESCAPE1(ANSI_CTRL_FAINT), footer);

  // Filtering can shrink the list, so erase rows left over from the previous
  // frame.
  frame += ANSI_CSI_ED(0);

  GetOutputStreamFileSP()->Lock() << frame;
}

void IOHandlerPicker::HandleKey(KeyPress press) {
  const size_t count = m_visible_indices.size();
  switch (press.key) {
  case Key::Up:
    if (m_cursor_index > 0)
      --m_cursor_index;
    break;
  case Key::Down:
    if (m_cursor_index + 1 < count)
      ++m_cursor_index;
    break;
  case Key::PageUp:
    m_cursor_index -= std::min(m_cursor_index, GetViewportHeight());
    break;
  case Key::PageDown:
    if (count)
      m_cursor_index =
          std::min(m_cursor_index + GetViewportHeight(), count - 1);
    break;
  case Key::Home:
    m_cursor_index = 0;
    break;
  case Key::End:
    if (count)
      m_cursor_index = count - 1;
    break;
  case Key::Enter:
    if (m_mode == PickerMode::SingleSelect) {
      if (!count)
        break;
      m_result.selected_indices.push_back(m_visible_indices[m_cursor_index]);
    } else {
      // Items hidden by the filter stay checked.
      for (unsigned i : m_checked.set_bits())
        m_result.selected_indices.push_back(i);
    }
    SetIsDone(true);
    break;
  case Key::Escape:
    Cancel();
    break;
  case Key::Char:
    switch (press.ch) {
    case 'k':
      HandleKey({Key::Up});
      break;
    case 'j':
      HandleKey({Key::Down});
      break;
    case 's':
    case 'S':
      CycleSort();
      break;
    case ' ':
      if (m_mode == PickerMode::MultiSelect && count)
        m_checked.flip(m_visible_indices[m_cursor_index]);
      break;
    case '/':
      m_searching = true;
      break;
    }
    break;
  case Key::Backspace:
  case Key::Abort:
  case Key::Wakeup:
  case Key::Unknown:
    break;
  }
}

void IOHandlerPicker::HandleSearchKey(KeyPress press) {
  switch (press.key) {
  case Key::Char:
    m_query += press.ch;
    UpdateVisibleIndices();
    break;
  case Key::Backspace:
    if (m_query.empty()) {
      m_searching = false;
      break;
    }
    m_query.pop_back();
    UpdateVisibleIndices();
    break;
  case Key::Enter:
    m_searching = false;
    break;
  case Key::Escape:
    m_searching = false;
    m_query.clear();
    UpdateVisibleIndices();
    break;
  default:
    // Arrow keys keep navigating the matches while typing.
    HandleKey(press);
    break;
  }
}

void IOHandlerPicker::Run() {
  if (!GetIsRealTerminal() || m_items.empty() || !EnterRawMode()) {
    m_result.was_canceled = true;
    SetIsDone(true);
    return;
  }

  UpdateLayout();

  // Hide the statusline before switching screens: its scroll region also
  // applies to the alternate screen, and its cursor save/restore shares the
  // slot that leaving the alternate screen restores.
  m_debugger.HideStatusline();

  // The alternate screen preserves the user's scrollback, and disabling
  // auto-wrap clips lines wider than measured instead of wrapping them.
  GetOutputStreamFileSP()->Lock()
      << ANSI_CSI_ENTER_ALT_SCREEN ANSI_CSI_HIDE_CURSOR
             ANSI_CSI_DISABLE_AUTO_WRAP;
  Render();

  while (!GetIsDone()) {
    KeyPress press = ReadKey();
    if (press.key == Key::Abort) {
      Cancel();
    } else if (press.key == Key::Wakeup) {
      // Unless Cancel() woke us up, the terminal was resized.
      if (!GetIsDone()) {
        UpdateLayout();
        // Rows may now be narrower than the previous frame, so wipe it.
        GetOutputStreamFileSP()->Lock() << ANSI_CSI_ERASE_VIEWPORT;
      }
    } else if (m_searching) {
      HandleSearchKey(press);
    } else {
      HandleKey(press);
    }

    if (!GetIsDone()) {
      ScrollToCursor();
      Render();
    }
  }

  GetOutputStreamFileSP()->Lock()
      << ANSI_CSI_ENABLE_AUTO_WRAP ANSI_CSI_SHOW_CURSOR
             ANSI_CSI_LEAVE_ALT_SCREEN;
  RestoreTerminalMode();
  m_debugger.ShowStatusline();
}

void IOHandlerPicker::Cancel() {
  // PopIOHandler() calls this after Run() returns. Keep its result.
  if (GetIsDone())
    return;
  m_result.was_canceled = true;
  SetIsDone(true);
  WakeUp();
}

bool IOHandlerPicker::Interrupt() {
  Cancel();
  return true;
}

void IOHandlerPicker::GotEOF() { Cancel(); }

void IOHandlerPicker::TerminalSizeChanged() { WakeUp(); }

void IOHandlerPicker::WakeUp() {
  if (!m_wakeup_pipe.CanWrite())
    return;
  char c = 0;
  if (llvm::Expected<size_t> bytes = m_wakeup_pipe.Write(&c, 1); !bytes)
    llvm::consumeError(bytes.takeError());
}
