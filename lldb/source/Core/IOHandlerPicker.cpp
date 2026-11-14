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

/// Reverse video gives the dialog a solid background in any color scheme.
static constexpr llvm::StringLiteral g_dialog_style =
    ANSI_ESCAPE1(ANSI_CTRL_IMAGE_NEGATIVE);
static constexpr llvm::StringLiteral g_cursor_style =
    ANSI_ESCAPE1(ANSI_CTRL_NORMAL);

/// Blank columns inside the dialog, on each side.
static constexpr size_t g_horizontal_padding = 1;
/// Fraction of the terminal's width left on each side of the dialog.
static constexpr size_t g_horizontal_margin_divisor = 10;

/// Blank line, header, key help, column names, footer, and blank line.
static constexpr size_t g_chrome_lines = 6;

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
                                 llvm::SmallVector<PickerColumnName> columns,
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

llvm::StringRef IOHandlerPicker::GetHelpText() const {
  return m_mode == PickerMode::SingleSelect
             ? "  jk: navigate  Enter: select  s: sort  /: search  ESC: cancel"
             : "  jk: navigate  Space: toggle  Enter: confirm  s: sort  "
               "/: search  ESC: cancel";
}

void IOHandlerPicker::ComputeColumnWidths() {
  m_column_widths.clear();
  size_t row_width = GetPrefixWidth();
  for (size_t i = 0; i < m_columns.size(); ++i) {
    size_t width = GetColumnWidth(m_columns[i]) + g_ascending_indicator.size();
    for (const PickerItem &item : m_items)
      width = std::max(width, GetColumnWidth(item.values[i]));
    m_column_widths.push_back(width);
    row_width += width + g_column_gap;
  }

  // Leave a margin on each side, unless the content needs that space.
  size_t content_width =
      std::max({row_width, GetColumnWidth(GetHelpText()),
                GetColumnWidth(m_header) + 2}) +
      2 * g_horizontal_padding;
  size_t margin = m_terminal_width / g_horizontal_margin_divisor;
  m_box_width = std::min(
      std::max(m_terminal_width - 2 * margin, content_width), m_terminal_width);

  size_t inner_width = GetInnerWidth();
  if (m_column_widths.empty() || row_width >= inner_width)
    return;

  // Spread the unused width across the columns.
  size_t num_cols = m_column_widths.size();
  size_t extra = inner_width - row_width;
  for (size_t i = 0; i < num_cols; ++i) {
    m_column_widths[i] += extra / num_cols;
    if (i < extra % num_cols)
      m_column_widths[i] += 1;
  }
}

size_t IOHandlerPicker::GetInnerWidth() const {
  size_t padding = 2 * g_horizontal_padding;
  return m_box_width > padding ? m_box_width - padding : 0;
}

void IOHandlerPicker::UpdateLayout() {
  m_terminal_width = m_debugger.GetTerminalWidth();
  m_terminal_height = m_debugger.GetTerminalHeight();
  ComputeColumnWidths();

  m_screen_height = m_terminal_height;
  if (m_debugger.StatuslineSupported() && m_screen_height > 1)
    --m_screen_height;

  // Keep a line of the screen visible above and below the dialog. Even on a
  // tiny terminal, keep the cursor's row in the dialog.
  size_t max_height =
      m_screen_height > 2 ? m_screen_height - 2 : m_screen_height;
  size_t max_rows =
      max_height > g_chrome_lines ? max_height - g_chrome_lines : 1;
  m_viewport_height = std::min(m_items.size(), max_rows);

  m_box_top = (m_screen_height - std::min(GetDialogHeight(), m_screen_height)) / 2;
  m_box_left = (m_terminal_width - m_box_width) / 2;
}

size_t IOHandlerPicker::GetDialogHeight() const {
  return m_viewport_height + g_chrome_lines;
}

bool IOHandlerPicker::EnterRawMode() {
#if LLDB_ENABLE_TERMIOS
  int fd = GetInputFD();
  if (fd < 0)
    return false;

  Terminal terminal(fd);
  // Without a saved state, we couldn't take the terminal out of raw mode.
  if (!m_saved_terminal_state.Save(terminal, /*save_process_group=*/false))
    return false;
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

void IOHandlerPicker::RestoreScreen() {
  // Only clear the lines above the statusline.
  std::string screen;
  for (size_t line = 1; line <= m_screen_height; ++line)
    screen += llvm::formatv(ANSI_ESC_START "{0};1H" ANSI_CSI_EL(2), line).str();
  screen += ANSI_CSI_RESET_CURSOR;

  // Leave room for the line the cursor ends up on.
  llvm::SmallVector<llvm::StringRef> lines;
  std::string history = m_debugger.GetTerminalHistory();
  llvm::StringRef(history).rtrim('\n').split(lines, '\n');
  size_t used = 0;
  size_t first = lines.size();
  while (first > 0) {
    size_t width = std::max<size_t>(GetColumnWidth(lines[first - 1]), 1);
    size_t height = m_terminal_width ? llvm::divideCeil(width, m_terminal_width)
                                     : 1;
    if (used + height >= m_screen_height)
      break;
    used += height;
    --first;
  }

  screen += ANSI_CSI_ENABLE_AUTO_WRAP;
  for (llvm::StringRef line : llvm::drop_begin(lines, first))
    screen += line.str() + "\r\n";
  screen += ANSI_CSI_DISABLE_AUTO_WRAP;
  GetOutputStreamFileSP()->Lock() << screen;
}

void IOHandlerPicker::Render() {
  // Write the frame at once, and put the cursor back where output goes.
  std::string frame = ANSI_ESC_SAVE_CURSOR;
  size_t line = m_box_top;
  auto move_to_next_line = [&]() {
    frame +=
        llvm::formatv(ANSI_ESC_START "{0};{1}H", ++line, m_box_left + 1).str();
  };
  size_t inner_width = GetInnerWidth();
  std::string padding(g_horizontal_padding, ' ');

  // Pad every line to the dialog's width so it overwrites the last frame.
  auto write_line = [&](llvm::StringRef style, llvm::StringRef content) {
    move_to_next_line();
    frame += g_dialog_style;
    frame += style;
    frame += padding;
    frame += ansi::TrimAndPad(content, inner_width);
    frame += padding;
    frame += ANSI_ESCAPE1(ANSI_CTRL_NORMAL);
  };

  write_line("", "");
  write_line(ANSI_ESCAPE1(ANSI_CTRL_BOLD), "  " + m_header);

  if (m_searching) {
    std::string prompt = "  /" + m_query + "_";
    prompt += "    Enter: apply  ESC: clear";
    write_line("", prompt);
  } else {
    write_line("", GetHelpText());
  }

  std::string column_names(GetPrefixWidth(), ' ');
  for (size_t i = 0; i < m_columns.size(); ++i) {
    std::string name = m_columns[i];
    bool is_sort_col = m_sort_order && m_sort_order->column == i;
    if (is_sort_col) {
      name += m_sort_order->ascending ? g_ascending_indicator
                                      : g_descending_indicator;
      column_names +=
          ANSI_ESCAPE1(ANSI_CTRL_BOLD) ANSI_ESCAPE1(ANSI_CTRL_UNDERLINE);
    }
    column_names += name;
    // Resetting the style also resets the dialog's, so restore it.
    if (is_sort_col) {
      column_names += ANSI_ESCAPE1(ANSI_CTRL_NORMAL);
      column_names += g_dialog_style;
    }
    column_names.append(
        m_column_widths[i] - GetColumnWidth(name) + g_column_gap, ' ');
  }
  write_line("", column_names);

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
    write_line(is_cursor ? g_cursor_style : "", row);
  }
  // Filtering can leave fewer rows than the dialog has room for.
  for (size_t i = end; i < m_viewport_start + GetViewportHeight(); ++i)
    write_line("", "");

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

  write_line("", "");
  frame += ANSI_ESC_RESTORE_CURSOR;

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
    m_result.was_canceled = true;
    SetIsDone(true);
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

  // Clip lines wider than measured instead of wrapping them.
  GetOutputStreamFileSP()->Lock()
      << ANSI_CSI_HIDE_CURSOR ANSI_CSI_DISABLE_AUTO_WRAP;
  Render();

  while (!GetIsDone()) {
    KeyPress press = ReadKey();
    if (m_cancel_requested || press.key == Key::Abort) {
      m_result.was_canceled = true;
      SetIsDone(true);
    } else if (press.key == Key::Wakeup) {
      // A resize, the statusline, or async output may have drawn over us.
      UpdateLayout();
      RestoreScreen();
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

  RestoreScreen();
  GetOutputStreamFileSP()->Lock()
      << ANSI_CSI_ENABLE_AUTO_WRAP ANSI_CSI_SHOW_CURSOR;
  RestoreTerminalMode();
}

void IOHandlerPicker::Cancel() {
  m_cancel_requested = true;
  WakeUp();
}

bool IOHandlerPicker::Interrupt() {
  Cancel();
  return true;
}

void IOHandlerPicker::GotEOF() { Cancel(); }

void IOHandlerPicker::TerminalSizeChanged() { WakeUp(); }

void IOHandlerPicker::Refresh() { WakeUp(); }

void IOHandlerPicker::PrintAsync(const char *s, size_t len, bool is_stdout) {
  // Debugger::PrintAsync() already added the text to the terminal history.
  WakeUp();
}

void IOHandlerPicker::WakeUp() {
  if (!m_wakeup_pipe.CanWrite())
    return;
  char c = 0;
  if (llvm::Expected<size_t> bytes = m_wakeup_pipe.Write(&c, 1); !bytes)
    llvm::consumeError(bytes.takeError());
}
