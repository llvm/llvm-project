//===-- IOHandlerCursesGUI.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/IOHandlerCursesGUI.h"
#include "lldb/Host/Config.h"

#if LLDB_ENABLE_CURSES
#if CURSES_HAVE_NCURSES_CURSES_H
#include <ncurses/curses.h>
#include <ncurses/panel.h>
#else
#include <curses.h>
#include <panel.h>
#endif
#endif

#if defined(__APPLE__)
#include <deque>
#endif
#include <string>

#include "lldb/Core/Debugger.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Host/File.h"
#include "lldb/Utility/Predicate.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StringList.h"
#include "lldb/lldb-forward.h"

#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"

#if LLDB_ENABLE_CURSES
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/State.h"
#endif

#include "llvm/ADT/StringRef.h"

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#endif

#include <memory>
#include <mutex>

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <type_traits>

using namespace lldb;
using namespace lldb_private;
using llvm::None;
using llvm::Optional;
using llvm::StringRef;

// we may want curses to be disabled for some builds for instance, windows
#if LLDB_ENABLE_CURSES

#define KEY_RETURN 10
#define KEY_ESCAPE 27

namespace curses {
class Menu;
class MenuDelegate;
class Window;
class WindowDelegate;
typedef std::shared_ptr<Menu> MenuSP;
typedef std::shared_ptr<MenuDelegate> MenuDelegateSP;
typedef std::shared_ptr<Window> WindowSP;
typedef std::shared_ptr<WindowDelegate> WindowDelegateSP;
typedef std::vector<MenuSP> Menus;
typedef std::vector<WindowSP> Windows;
typedef std::vector<WindowDelegateSP> WindowDelegates;

#if 0
type summary add -s "x=${var.x}, y=${var.y}" curses::Point
type summary add -s "w=${var.width}, h=${var.height}" curses::Size
type summary add -s "${var.origin%S} ${var.size%S}" curses::Rect
#endif

struct Point {
  int x;
  int y;

  Point(int _x = 0, int _y = 0) : x(_x), y(_y) {}

  void Clear() {
    x = 0;
    y = 0;
  }

  Point &operator+=(const Point &rhs) {
    x += rhs.x;
    y += rhs.y;
    return *this;
  }

  void Dump() { printf("(x=%i, y=%i)\n", x, y); }
};

bool operator==(const Point &lhs, const Point &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool operator!=(const Point &lhs, const Point &rhs) {
  return lhs.x != rhs.x || lhs.y != rhs.y;
}

struct Size {
  int width;
  int height;
  Size(int w = 0, int h = 0) : width(w), height(h) {}

  void Clear() {
    width = 0;
    height = 0;
  }

  void Dump() { printf("(w=%i, h=%i)\n", width, height); }
};

bool operator==(const Size &lhs, const Size &rhs) {
  return lhs.width == rhs.width && lhs.height == rhs.height;
}

bool operator!=(const Size &lhs, const Size &rhs) {
  return lhs.width != rhs.width || lhs.height != rhs.height;
}

struct Rect {
  Point origin;
  Size size;

  Rect() : origin(), size() {}

  Rect(const Point &p, const Size &s) : origin(p), size(s) {}

  void Clear() {
    origin.Clear();
    size.Clear();
  }

  void Dump() {
    printf("(x=%i, y=%i), w=%i, h=%i)\n", origin.x, origin.y, size.width,
           size.height);
  }

  void Inset(int w, int h) {
    if (size.width > w * 2)
      size.width -= w * 2;
    origin.x += w;

    if (size.height > h * 2)
      size.height -= h * 2;
    origin.y += h;
  }

  // Return a status bar rectangle which is the last line of this rectangle.
  // This rectangle will be modified to not include the status bar area.
  Rect MakeStatusBar() {
    Rect status_bar;
    if (size.height > 1) {
      status_bar.origin.x = origin.x;
      status_bar.origin.y = size.height;
      status_bar.size.width = size.width;
      status_bar.size.height = 1;
      --size.height;
    }
    return status_bar;
  }

  // Return a menubar rectangle which is the first line of this rectangle. This
  // rectangle will be modified to not include the menubar area.
  Rect MakeMenuBar() {
    Rect menubar;
    if (size.height > 1) {
      menubar.origin.x = origin.x;
      menubar.origin.y = origin.y;
      menubar.size.width = size.width;
      menubar.size.height = 1;
      ++origin.y;
      --size.height;
    }
    return menubar;
  }

  void HorizontalSplitPercentage(float top_percentage, Rect &top,
                                 Rect &bottom) const {
    float top_height = top_percentage * size.height;
    HorizontalSplit(top_height, top, bottom);
  }

  void HorizontalSplit(int top_height, Rect &top, Rect &bottom) const {
    top = *this;
    if (top_height < size.height) {
      top.size.height = top_height;
      bottom.origin.x = origin.x;
      bottom.origin.y = origin.y + top.size.height;
      bottom.size.width = size.width;
      bottom.size.height = size.height - top.size.height;
    } else {
      bottom.Clear();
    }
  }

  void VerticalSplitPercentage(float left_percentage, Rect &left,
                               Rect &right) const {
    float left_width = left_percentage * size.width;
    VerticalSplit(left_width, left, right);
  }

  void VerticalSplit(int left_width, Rect &left, Rect &right) const {
    left = *this;
    if (left_width < size.width) {
      left.size.width = left_width;
      right.origin.x = origin.x + left.size.width;
      right.origin.y = origin.y;
      right.size.width = size.width - left.size.width;
      right.size.height = size.height;
    } else {
      right.Clear();
    }
  }
};

bool operator==(const Rect &lhs, const Rect &rhs) {
  return lhs.origin == rhs.origin && lhs.size == rhs.size;
}

bool operator!=(const Rect &lhs, const Rect &rhs) {
  return lhs.origin != rhs.origin || lhs.size != rhs.size;
}

enum HandleCharResult {
  eKeyNotHandled = 0,
  eKeyHandled = 1,
  eQuitApplication = 2
};

enum class MenuActionResult {
  Handled,
  NotHandled,
  Quit // Exit all menus and quit
};

struct KeyHelp {
  int ch;
  const char *description;
};

// COLOR_PAIR index names
enum {
  // First 16 colors are 8 black background and 8 blue background colors,
  // needed by OutputColoredStringTruncated().
  BlackOnBlack = 1,
  RedOnBlack,
  GreenOnBlack,
  YellowOnBlack,
  BlueOnBlack,
  MagentaOnBlack,
  CyanOnBlack,
  WhiteOnBlack,
  BlackOnBlue,
  RedOnBlue,
  GreenOnBlue,
  YellowOnBlue,
  BlueOnBlue,
  MagentaOnBlue,
  CyanOnBlue,
  WhiteOnBlue,
  // Other colors, as needed.
  BlackOnWhite,
  MagentaOnWhite,
  LastColorPairIndex = MagentaOnWhite
};

class WindowDelegate {
public:
  virtual ~WindowDelegate() = default;

  virtual bool WindowDelegateDraw(Window &window, bool force) {
    return false; // Drawing not handled
  }

  virtual HandleCharResult WindowDelegateHandleChar(Window &window, int key) {
    return eKeyNotHandled;
  }

  virtual const char *WindowDelegateGetHelpText() { return nullptr; }

  virtual KeyHelp *WindowDelegateGetKeyHelp() { return nullptr; }
};

class HelpDialogDelegate : public WindowDelegate {
public:
  HelpDialogDelegate(const char *text, KeyHelp *key_help_array);

  ~HelpDialogDelegate() override;

  bool WindowDelegateDraw(Window &window, bool force) override;

  HandleCharResult WindowDelegateHandleChar(Window &window, int key) override;

  size_t GetNumLines() const { return m_text.GetSize(); }

  size_t GetMaxLineLength() const { return m_text.GetMaxStringLength(); }

protected:
  StringList m_text;
  int m_first_visible_line;
};

class Window {
public:
  Window(const char *name)
      : m_name(name), m_window(nullptr), m_panel(nullptr), m_parent(nullptr),
        m_subwindows(), m_delegate_sp(), m_curr_active_window_idx(UINT32_MAX),
        m_prev_active_window_idx(UINT32_MAX), m_delete(false),
        m_needs_update(true), m_can_activate(true), m_is_subwin(false) {}

  Window(const char *name, WINDOW *w, bool del = true)
      : m_name(name), m_window(nullptr), m_panel(nullptr), m_parent(nullptr),
        m_subwindows(), m_delegate_sp(), m_curr_active_window_idx(UINT32_MAX),
        m_prev_active_window_idx(UINT32_MAX), m_delete(del),
        m_needs_update(true), m_can_activate(true), m_is_subwin(false) {
    if (w)
      Reset(w);
  }

  Window(const char *name, const Rect &bounds)
      : m_name(name), m_window(nullptr), m_parent(nullptr), m_subwindows(),
        m_delegate_sp(), m_curr_active_window_idx(UINT32_MAX),
        m_prev_active_window_idx(UINT32_MAX), m_delete(true),
        m_needs_update(true), m_can_activate(true), m_is_subwin(false) {
    Reset(::newwin(bounds.size.height, bounds.size.width, bounds.origin.y,
                   bounds.origin.y));
  }

  virtual ~Window() {
    RemoveSubWindows();
    Reset();
  }

  void Reset(WINDOW *w = nullptr, bool del = true) {
    if (m_window == w)
      return;

    if (m_panel) {
      ::del_panel(m_panel);
      m_panel = nullptr;
    }
    if (m_window && m_delete) {
      ::delwin(m_window);
      m_window = nullptr;
      m_delete = false;
    }
    if (w) {
      m_window = w;
      m_panel = ::new_panel(m_window);
      m_delete = del;
    }
  }

  void AttributeOn(attr_t attr) { ::wattron(m_window, attr); }
  void AttributeOff(attr_t attr) { ::wattroff(m_window, attr); }
  void Box(chtype v_char = ACS_VLINE, chtype h_char = ACS_HLINE) {
    ::box(m_window, v_char, h_char);
  }
  void Clear() { ::wclear(m_window); }
  void Erase() { ::werase(m_window); }
  Rect GetBounds() const {
    return Rect(GetParentOrigin(), GetSize());
  } // Get the rectangle in our parent window
  int GetChar() { return ::wgetch(m_window); }
  int GetCursorX() const { return getcurx(m_window); }
  int GetCursorY() const { return getcury(m_window); }
  Rect GetFrame() const {
    return Rect(Point(), GetSize());
  } // Get our rectangle in our own coordinate system
  Point GetParentOrigin() const { return Point(GetParentX(), GetParentY()); }
  Size GetSize() const { return Size(GetWidth(), GetHeight()); }
  int GetParentX() const { return getparx(m_window); }
  int GetParentY() const { return getpary(m_window); }
  int GetMaxX() const { return getmaxx(m_window); }
  int GetMaxY() const { return getmaxy(m_window); }
  int GetWidth() const { return GetMaxX(); }
  int GetHeight() const { return GetMaxY(); }
  void MoveCursor(int x, int y) { ::wmove(m_window, y, x); }
  void MoveWindow(int x, int y) { MoveWindow(Point(x, y)); }
  void Resize(int w, int h) { ::wresize(m_window, h, w); }
  void Resize(const Size &size) {
    ::wresize(m_window, size.height, size.width);
  }
  void PutChar(int ch) { ::waddch(m_window, ch); }
  void PutCString(const char *s, int len = -1) { ::waddnstr(m_window, s, len); }
  void SetBackground(int color_pair_idx) {
    ::wbkgd(m_window, COLOR_PAIR(color_pair_idx));
  }

  void PutCStringTruncated(int right_pad, const char *s, int len = -1) {
    int bytes_left = GetWidth() - GetCursorX();
    if (bytes_left > right_pad) {
      bytes_left -= right_pad;
      ::waddnstr(m_window, s, len < 0 ? bytes_left : std::min(bytes_left, len));
    }
  }

  void MoveWindow(const Point &origin) {
    const bool moving_window = origin != GetParentOrigin();
    if (m_is_subwin && moving_window) {
      // Can't move subwindows, must delete and re-create
      Size size = GetSize();
      Reset(::subwin(m_parent->m_window, size.height, size.width, origin.y,
                     origin.x),
            true);
    } else {
      ::mvwin(m_window, origin.y, origin.x);
    }
  }

  void SetBounds(const Rect &bounds) {
    const bool moving_window = bounds.origin != GetParentOrigin();
    if (m_is_subwin && moving_window) {
      // Can't move subwindows, must delete and re-create
      Reset(::subwin(m_parent->m_window, bounds.size.height, bounds.size.width,
                     bounds.origin.y, bounds.origin.x),
            true);
    } else {
      if (moving_window)
        MoveWindow(bounds.origin);
      Resize(bounds.size);
    }
  }

  void Printf(const char *format, ...) __attribute__((format(printf, 2, 3))) {
    va_list args;
    va_start(args, format);
    vwprintw(m_window, format, args);
    va_end(args);
  }

  void PrintfTruncated(int right_pad, const char *format, ...)
      __attribute__((format(printf, 3, 4))) {
    va_list args;
    va_start(args, format);
    StreamString strm;
    strm.PrintfVarArg(format, args);
    va_end(args);
    PutCStringTruncated(right_pad, strm.GetData());
  }

  size_t LimitLengthToRestOfLine(size_t length) const {
    return std::min<size_t>(length, std::max(0, GetWidth() - GetCursorX() - 1));
  }

  // Curses doesn't allow direct output of color escape sequences, but that's
  // how we get source lines from the Highligher class. Read the line and
  // convert color escape sequences to curses color attributes.
  void OutputColoredStringTruncated(int right_pad, StringRef string,
                                    bool use_blue_background) {
    attr_t saved_attr;
    short saved_pair;
    wattr_get(m_window, &saved_attr, &saved_pair, nullptr);
    if (use_blue_background)
      ::wattron(m_window, COLOR_PAIR(WhiteOnBlue));
    while (!string.empty()) {
      size_t esc_pos = string.find('\x1b');
      if (esc_pos == StringRef::npos) {
        PutCStringTruncated(right_pad, string.data(), string.size());
        break;
      }
      if (esc_pos > 0) {
        PutCStringTruncated(right_pad, string.data(), esc_pos);
        string = string.drop_front(esc_pos);
      }
      bool consumed = string.consume_front("\x1b");
      assert(consumed);
      UNUSED_IF_ASSERT_DISABLED(consumed);
      // This is written to match our Highlighter classes, which seem to
      // generate only foreground color escape sequences. If necessary, this
      // will need to be extended.
      if (!string.consume_front("[")) {
        llvm::errs() << "Missing '[' in color escape sequence.\n";
        continue;
      }
      // Only 8 basic foreground colors and reset, our Highlighter doesn't use
      // anything else.
      int value;
      if (!!string.consumeInteger(10, value) || // Returns false on success.
          !(value == 0 || (value >= 30 && value <= 37))) {
        llvm::errs() << "No valid color code in color escape sequence.\n";
        continue;
      }
      if (!string.consume_front("m")) {
        llvm::errs() << "Missing 'm' in color escape sequence.\n";
        continue;
      }
      if (value == 0) { // Reset.
        wattr_set(m_window, saved_attr, saved_pair, nullptr);
        if (use_blue_background)
          ::wattron(m_window, COLOR_PAIR(WhiteOnBlue));
      } else {
        // Mapped directly to first 16 color pairs (black/blue background).
        ::wattron(m_window,
                  COLOR_PAIR(value - 30 + 1 + (use_blue_background ? 8 : 0)));
      }
    }
    wattr_set(m_window, saved_attr, saved_pair, nullptr);
  }

  void Touch() {
    ::touchwin(m_window);
    if (m_parent)
      m_parent->Touch();
  }

  WindowSP CreateSubWindow(const char *name, const Rect &bounds,
                           bool make_active) {
    auto get_window = [this, &bounds]() {
      return m_window
                 ? ::subwin(m_window, bounds.size.height, bounds.size.width,
                            bounds.origin.y, bounds.origin.x)
                 : ::newwin(bounds.size.height, bounds.size.width,
                            bounds.origin.y, bounds.origin.x);
    };
    WindowSP subwindow_sp = std::make_shared<Window>(name, get_window(), true);
    subwindow_sp->m_is_subwin = subwindow_sp.operator bool();
    subwindow_sp->m_parent = this;
    if (make_active) {
      m_prev_active_window_idx = m_curr_active_window_idx;
      m_curr_active_window_idx = m_subwindows.size();
    }
    m_subwindows.push_back(subwindow_sp);
    ::top_panel(subwindow_sp->m_panel);
    m_needs_update = true;
    return subwindow_sp;
  }

  bool RemoveSubWindow(Window *window) {
    Windows::iterator pos, end = m_subwindows.end();
    size_t i = 0;
    for (pos = m_subwindows.begin(); pos != end; ++pos, ++i) {
      if ((*pos).get() == window) {
        if (m_prev_active_window_idx == i)
          m_prev_active_window_idx = UINT32_MAX;
        else if (m_prev_active_window_idx != UINT32_MAX &&
                 m_prev_active_window_idx > i)
          --m_prev_active_window_idx;

        if (m_curr_active_window_idx == i)
          m_curr_active_window_idx = UINT32_MAX;
        else if (m_curr_active_window_idx != UINT32_MAX &&
                 m_curr_active_window_idx > i)
          --m_curr_active_window_idx;
        window->Erase();
        m_subwindows.erase(pos);
        m_needs_update = true;
        if (m_parent)
          m_parent->Touch();
        else
          ::touchwin(stdscr);
        return true;
      }
    }
    return false;
  }

  WindowSP FindSubWindow(const char *name) {
    Windows::iterator pos, end = m_subwindows.end();
    size_t i = 0;
    for (pos = m_subwindows.begin(); pos != end; ++pos, ++i) {
      if ((*pos)->m_name == name)
        return *pos;
    }
    return WindowSP();
  }

  void RemoveSubWindows() {
    m_curr_active_window_idx = UINT32_MAX;
    m_prev_active_window_idx = UINT32_MAX;
    for (Windows::iterator pos = m_subwindows.begin();
         pos != m_subwindows.end(); pos = m_subwindows.erase(pos)) {
      (*pos)->Erase();
    }
    if (m_parent)
      m_parent->Touch();
    else
      ::touchwin(stdscr);
  }

  WINDOW *get() { return m_window; }

  operator WINDOW *() { return m_window; }

  // Window drawing utilities
  void DrawTitleBox(const char *title, const char *bottom_message = nullptr) {
    attr_t attr = 0;
    if (IsActive())
      attr = A_BOLD | COLOR_PAIR(BlackOnWhite);
    else
      attr = 0;
    if (attr)
      AttributeOn(attr);

    Box();
    MoveCursor(3, 0);

    if (title && title[0]) {
      PutChar('<');
      PutCString(title);
      PutChar('>');
    }

    if (bottom_message && bottom_message[0]) {
      int bottom_message_length = strlen(bottom_message);
      int x = GetWidth() - 3 - (bottom_message_length + 2);

      if (x > 0) {
        MoveCursor(x, GetHeight() - 1);
        PutChar('[');
        PutCString(bottom_message);
        PutChar(']');
      } else {
        MoveCursor(1, GetHeight() - 1);
        PutChar('[');
        PutCStringTruncated(1, bottom_message);
      }
    }
    if (attr)
      AttributeOff(attr);
  }

  virtual void Draw(bool force) {
    if (m_delegate_sp && m_delegate_sp->WindowDelegateDraw(*this, force))
      return;

    for (auto &subwindow_sp : m_subwindows)
      subwindow_sp->Draw(force);
  }

  bool CreateHelpSubwindow() {
    if (m_delegate_sp) {
      const char *text = m_delegate_sp->WindowDelegateGetHelpText();
      KeyHelp *key_help = m_delegate_sp->WindowDelegateGetKeyHelp();
      if ((text && text[0]) || key_help) {
        std::unique_ptr<HelpDialogDelegate> help_delegate_up(
            new HelpDialogDelegate(text, key_help));
        const size_t num_lines = help_delegate_up->GetNumLines();
        const size_t max_length = help_delegate_up->GetMaxLineLength();
        Rect bounds = GetBounds();
        bounds.Inset(1, 1);
        if (max_length + 4 < static_cast<size_t>(bounds.size.width)) {
          bounds.origin.x += (bounds.size.width - max_length + 4) / 2;
          bounds.size.width = max_length + 4;
        } else {
          if (bounds.size.width > 100) {
            const int inset_w = bounds.size.width / 4;
            bounds.origin.x += inset_w;
            bounds.size.width -= 2 * inset_w;
          }
        }

        if (num_lines + 2 < static_cast<size_t>(bounds.size.height)) {
          bounds.origin.y += (bounds.size.height - num_lines + 2) / 2;
          bounds.size.height = num_lines + 2;
        } else {
          if (bounds.size.height > 100) {
            const int inset_h = bounds.size.height / 4;
            bounds.origin.y += inset_h;
            bounds.size.height -= 2 * inset_h;
          }
        }
        WindowSP help_window_sp;
        Window *parent_window = GetParent();
        if (parent_window)
          help_window_sp = parent_window->CreateSubWindow("Help", bounds, true);
        else
          help_window_sp = CreateSubWindow("Help", bounds, true);
        help_window_sp->SetDelegate(
            WindowDelegateSP(help_delegate_up.release()));
        return true;
      }
    }
    return false;
  }

  virtual HandleCharResult HandleChar(int key) {
    // Always check the active window first
    HandleCharResult result = eKeyNotHandled;
    WindowSP active_window_sp = GetActiveWindow();
    if (active_window_sp) {
      result = active_window_sp->HandleChar(key);
      if (result != eKeyNotHandled)
        return result;
    }

    if (m_delegate_sp) {
      result = m_delegate_sp->WindowDelegateHandleChar(*this, key);
      if (result != eKeyNotHandled)
        return result;
    }

    // Then check for any windows that want any keys that weren't handled. This
    // is typically only for a menubar. Make a copy of the subwindows in case
    // any HandleChar() functions muck with the subwindows. If we don't do
    // this, we can crash when iterating over the subwindows.
    Windows subwindows(m_subwindows);
    for (auto subwindow_sp : subwindows) {
      if (!subwindow_sp->m_can_activate) {
        HandleCharResult result = subwindow_sp->HandleChar(key);
        if (result != eKeyNotHandled)
          return result;
      }
    }

    return eKeyNotHandled;
  }

  WindowSP GetActiveWindow() {
    if (!m_subwindows.empty()) {
      if (m_curr_active_window_idx >= m_subwindows.size()) {
        if (m_prev_active_window_idx < m_subwindows.size()) {
          m_curr_active_window_idx = m_prev_active_window_idx;
          m_prev_active_window_idx = UINT32_MAX;
        } else if (IsActive()) {
          m_prev_active_window_idx = UINT32_MAX;
          m_curr_active_window_idx = UINT32_MAX;

          // Find first window that wants to be active if this window is active
          const size_t num_subwindows = m_subwindows.size();
          for (size_t i = 0; i < num_subwindows; ++i) {
            if (m_subwindows[i]->GetCanBeActive()) {
              m_curr_active_window_idx = i;
              break;
            }
          }
        }
      }

      if (m_curr_active_window_idx < m_subwindows.size())
        return m_subwindows[m_curr_active_window_idx];
    }
    return WindowSP();
  }

  bool GetCanBeActive() const { return m_can_activate; }

  void SetCanBeActive(bool b) { m_can_activate = b; }

  void SetDelegate(const WindowDelegateSP &delegate_sp) {
    m_delegate_sp = delegate_sp;
  }

  Window *GetParent() const { return m_parent; }

  bool IsActive() const {
    if (m_parent)
      return m_parent->GetActiveWindow().get() == this;
    else
      return true; // Top level window is always active
  }

  void SelectNextWindowAsActive() {
    // Move active focus to next window
    const int num_subwindows = m_subwindows.size();
    int start_idx = 0;
    if (m_curr_active_window_idx != UINT32_MAX) {
      m_prev_active_window_idx = m_curr_active_window_idx;
      start_idx = m_curr_active_window_idx + 1;
    }
    for (int idx = start_idx; idx < num_subwindows; ++idx) {
      if (m_subwindows[idx]->GetCanBeActive()) {
        m_curr_active_window_idx = idx;
        return;
      }
    }
    for (int idx = 0; idx < start_idx; ++idx) {
      if (m_subwindows[idx]->GetCanBeActive()) {
        m_curr_active_window_idx = idx;
        break;
      }
    }
  }

  void SelectPreviousWindowAsActive() {
    // Move active focus to previous window
    const int num_subwindows = m_subwindows.size();
    int start_idx = num_subwindows - 1;
    if (m_curr_active_window_idx != UINT32_MAX) {
      m_prev_active_window_idx = m_curr_active_window_idx;
      start_idx = m_curr_active_window_idx - 1;
    }
    for (int idx = start_idx; idx >= 0; --idx) {
      if (m_subwindows[idx]->GetCanBeActive()) {
        m_curr_active_window_idx = idx;
        return;
      }
    }
    for (int idx = num_subwindows - 1; idx > start_idx; --idx) {
      if (m_subwindows[idx]->GetCanBeActive()) {
        m_curr_active_window_idx = idx;
        break;
      }
    }
  }

  const char *GetName() const { return m_name.c_str(); }

protected:
  std::string m_name;
  WINDOW *m_window;
  PANEL *m_panel;
  Window *m_parent;
  Windows m_subwindows;
  WindowDelegateSP m_delegate_sp;
  uint32_t m_curr_active_window_idx;
  uint32_t m_prev_active_window_idx;
  bool m_delete;
  bool m_needs_update;
  bool m_can_activate;
  bool m_is_subwin;

private:
  Window(const Window &) = delete;
  const Window &operator=(const Window &) = delete;
};

class MenuDelegate {
public:
  virtual ~MenuDelegate() = default;

  virtual MenuActionResult MenuDelegateAction(Menu &menu) = 0;
};

class Menu : public WindowDelegate {
public:
  enum class Type { Invalid, Bar, Item, Separator };

  // Menubar or separator constructor
  Menu(Type type);

  // Menuitem constructor
  Menu(const char *name, const char *key_name, int key_value,
       uint64_t identifier);

  ~Menu() override = default;

  const MenuDelegateSP &GetDelegate() const { return m_delegate_sp; }

  void SetDelegate(const MenuDelegateSP &delegate_sp) {
    m_delegate_sp = delegate_sp;
  }

  void RecalculateNameLengths();

  void AddSubmenu(const MenuSP &menu_sp);

  int DrawAndRunMenu(Window &window);

  void DrawMenuTitle(Window &window, bool highlight);

  bool WindowDelegateDraw(Window &window, bool force) override;

  HandleCharResult WindowDelegateHandleChar(Window &window, int key) override;

  MenuActionResult ActionPrivate(Menu &menu) {
    MenuActionResult result = MenuActionResult::NotHandled;
    if (m_delegate_sp) {
      result = m_delegate_sp->MenuDelegateAction(menu);
      if (result != MenuActionResult::NotHandled)
        return result;
    } else if (m_parent) {
      result = m_parent->ActionPrivate(menu);
      if (result != MenuActionResult::NotHandled)
        return result;
    }
    return m_canned_result;
  }

  MenuActionResult Action() {
    // Call the recursive action so it can try to handle it with the menu
    // delegate, and if not, try our parent menu
    return ActionPrivate(*this);
  }

  void SetCannedResult(MenuActionResult result) { m_canned_result = result; }

  Menus &GetSubmenus() { return m_submenus; }

  const Menus &GetSubmenus() const { return m_submenus; }

  int GetSelectedSubmenuIndex() const { return m_selected; }

  void SetSelectedSubmenuIndex(int idx) { m_selected = idx; }

  Type GetType() const { return m_type; }

  int GetStartingColumn() const { return m_start_col; }

  void SetStartingColumn(int col) { m_start_col = col; }

  int GetKeyValue() const { return m_key_value; }

  std::string &GetName() { return m_name; }

  int GetDrawWidth() const {
    return m_max_submenu_name_length + m_max_submenu_key_name_length + 8;
  }

  uint64_t GetIdentifier() const { return m_identifier; }

  void SetIdentifier(uint64_t identifier) { m_identifier = identifier; }

protected:
  std::string m_name;
  std::string m_key_name;
  uint64_t m_identifier;
  Type m_type;
  int m_key_value;
  int m_start_col;
  int m_max_submenu_name_length;
  int m_max_submenu_key_name_length;
  int m_selected;
  Menu *m_parent;
  Menus m_submenus;
  WindowSP m_menu_window_sp;
  MenuActionResult m_canned_result;
  MenuDelegateSP m_delegate_sp;
};

// Menubar or separator constructor
Menu::Menu(Type type)
    : m_name(), m_key_name(), m_identifier(0), m_type(type), m_key_value(0),
      m_start_col(0), m_max_submenu_name_length(0),
      m_max_submenu_key_name_length(0), m_selected(0), m_parent(nullptr),
      m_submenus(), m_canned_result(MenuActionResult::NotHandled),
      m_delegate_sp() {}

// Menuitem constructor
Menu::Menu(const char *name, const char *key_name, int key_value,
           uint64_t identifier)
    : m_name(), m_key_name(), m_identifier(identifier), m_type(Type::Invalid),
      m_key_value(key_value), m_start_col(0), m_max_submenu_name_length(0),
      m_max_submenu_key_name_length(0), m_selected(0), m_parent(nullptr),
      m_submenus(), m_canned_result(MenuActionResult::NotHandled),
      m_delegate_sp() {
  if (name && name[0]) {
    m_name = name;
    m_type = Type::Item;
    if (key_name && key_name[0])
      m_key_name = key_name;
  } else {
    m_type = Type::Separator;
  }
}

void Menu::RecalculateNameLengths() {
  m_max_submenu_name_length = 0;
  m_max_submenu_key_name_length = 0;
  Menus &submenus = GetSubmenus();
  const size_t num_submenus = submenus.size();
  for (size_t i = 0; i < num_submenus; ++i) {
    Menu *submenu = submenus[i].get();
    if (static_cast<size_t>(m_max_submenu_name_length) < submenu->m_name.size())
      m_max_submenu_name_length = submenu->m_name.size();
    if (static_cast<size_t>(m_max_submenu_key_name_length) <
        submenu->m_key_name.size())
      m_max_submenu_key_name_length = submenu->m_key_name.size();
  }
}

void Menu::AddSubmenu(const MenuSP &menu_sp) {
  menu_sp->m_parent = this;
  if (static_cast<size_t>(m_max_submenu_name_length) < menu_sp->m_name.size())
    m_max_submenu_name_length = menu_sp->m_name.size();
  if (static_cast<size_t>(m_max_submenu_key_name_length) <
      menu_sp->m_key_name.size())
    m_max_submenu_key_name_length = menu_sp->m_key_name.size();
  m_submenus.push_back(menu_sp);
}

void Menu::DrawMenuTitle(Window &window, bool highlight) {
  if (m_type == Type::Separator) {
    window.MoveCursor(0, window.GetCursorY());
    window.PutChar(ACS_LTEE);
    int width = window.GetWidth();
    if (width > 2) {
      width -= 2;
      for (int i = 0; i < width; ++i)
        window.PutChar(ACS_HLINE);
    }
    window.PutChar(ACS_RTEE);
  } else {
    const int shortcut_key = m_key_value;
    bool underlined_shortcut = false;
    const attr_t highlight_attr = A_REVERSE;
    if (highlight)
      window.AttributeOn(highlight_attr);
    if (llvm::isPrint(shortcut_key)) {
      size_t lower_pos = m_name.find(tolower(shortcut_key));
      size_t upper_pos = m_name.find(toupper(shortcut_key));
      const char *name = m_name.c_str();
      size_t pos = std::min<size_t>(lower_pos, upper_pos);
      if (pos != std::string::npos) {
        underlined_shortcut = true;
        if (pos > 0) {
          window.PutCString(name, pos);
          name += pos;
        }
        const attr_t shortcut_attr = A_UNDERLINE | A_BOLD;
        window.AttributeOn(shortcut_attr);
        window.PutChar(name[0]);
        window.AttributeOff(shortcut_attr);
        name++;
        if (name[0])
          window.PutCString(name);
      }
    }

    if (!underlined_shortcut) {
      window.PutCString(m_name.c_str());
    }

    if (highlight)
      window.AttributeOff(highlight_attr);

    if (m_key_name.empty()) {
      if (!underlined_shortcut && llvm::isPrint(m_key_value)) {
        window.AttributeOn(COLOR_PAIR(MagentaOnWhite));
        window.Printf(" (%c)", m_key_value);
        window.AttributeOff(COLOR_PAIR(MagentaOnWhite));
      }
    } else {
      window.AttributeOn(COLOR_PAIR(MagentaOnWhite));
      window.Printf(" (%s)", m_key_name.c_str());
      window.AttributeOff(COLOR_PAIR(MagentaOnWhite));
    }
  }
}

bool Menu::WindowDelegateDraw(Window &window, bool force) {
  Menus &submenus = GetSubmenus();
  const size_t num_submenus = submenus.size();
  const int selected_idx = GetSelectedSubmenuIndex();
  Menu::Type menu_type = GetType();
  switch (menu_type) {
  case Menu::Type::Bar: {
    window.SetBackground(BlackOnWhite);
    window.MoveCursor(0, 0);
    for (size_t i = 0; i < num_submenus; ++i) {
      Menu *menu = submenus[i].get();
      if (i > 0)
        window.PutChar(' ');
      menu->SetStartingColumn(window.GetCursorX());
      window.PutCString("| ");
      menu->DrawMenuTitle(window, false);
    }
    window.PutCString(" |");
  } break;

  case Menu::Type::Item: {
    int y = 1;
    int x = 3;
    // Draw the menu
    int cursor_x = 0;
    int cursor_y = 0;
    window.Erase();
    window.SetBackground(BlackOnWhite);
    window.Box();
    for (size_t i = 0; i < num_submenus; ++i) {
      const bool is_selected = (i == static_cast<size_t>(selected_idx));
      window.MoveCursor(x, y + i);
      if (is_selected) {
        // Remember where we want the cursor to be
        cursor_x = x - 1;
        cursor_y = y + i;
      }
      submenus[i]->DrawMenuTitle(window, is_selected);
    }
    window.MoveCursor(cursor_x, cursor_y);
  } break;

  default:
  case Menu::Type::Separator:
    break;
  }
  return true; // Drawing handled...
}

HandleCharResult Menu::WindowDelegateHandleChar(Window &window, int key) {
  HandleCharResult result = eKeyNotHandled;

  Menus &submenus = GetSubmenus();
  const size_t num_submenus = submenus.size();
  const int selected_idx = GetSelectedSubmenuIndex();
  Menu::Type menu_type = GetType();
  if (menu_type == Menu::Type::Bar) {
    MenuSP run_menu_sp;
    switch (key) {
    case KEY_DOWN:
    case KEY_UP:
      // Show last menu or first menu
      if (selected_idx < static_cast<int>(num_submenus))
        run_menu_sp = submenus[selected_idx];
      else if (!submenus.empty())
        run_menu_sp = submenus.front();
      result = eKeyHandled;
      break;

    case KEY_RIGHT:
      ++m_selected;
      if (m_selected >= static_cast<int>(num_submenus))
        m_selected = 0;
      if (m_selected < static_cast<int>(num_submenus))
        run_menu_sp = submenus[m_selected];
      else if (!submenus.empty())
        run_menu_sp = submenus.front();
      result = eKeyHandled;
      break;

    case KEY_LEFT:
      --m_selected;
      if (m_selected < 0)
        m_selected = num_submenus - 1;
      if (m_selected < static_cast<int>(num_submenus))
        run_menu_sp = submenus[m_selected];
      else if (!submenus.empty())
        run_menu_sp = submenus.front();
      result = eKeyHandled;
      break;

    default:
      for (size_t i = 0; i < num_submenus; ++i) {
        if (submenus[i]->GetKeyValue() == key) {
          SetSelectedSubmenuIndex(i);
          run_menu_sp = submenus[i];
          result = eKeyHandled;
          break;
        }
      }
      break;
    }

    if (run_menu_sp) {
      // Run the action on this menu in case we need to populate the menu with
      // dynamic content and also in case check marks, and any other menu
      // decorations need to be calculated
      if (run_menu_sp->Action() == MenuActionResult::Quit)
        return eQuitApplication;

      Rect menu_bounds;
      menu_bounds.origin.x = run_menu_sp->GetStartingColumn();
      menu_bounds.origin.y = 1;
      menu_bounds.size.width = run_menu_sp->GetDrawWidth();
      menu_bounds.size.height = run_menu_sp->GetSubmenus().size() + 2;
      if (m_menu_window_sp)
        window.GetParent()->RemoveSubWindow(m_menu_window_sp.get());

      m_menu_window_sp = window.GetParent()->CreateSubWindow(
          run_menu_sp->GetName().c_str(), menu_bounds, true);
      m_menu_window_sp->SetDelegate(run_menu_sp);
    }
  } else if (menu_type == Menu::Type::Item) {
    switch (key) {
    case KEY_DOWN:
      if (m_submenus.size() > 1) {
        const int start_select = m_selected;
        while (++m_selected != start_select) {
          if (static_cast<size_t>(m_selected) >= num_submenus)
            m_selected = 0;
          if (m_submenus[m_selected]->GetType() == Type::Separator)
            continue;
          else
            break;
        }
        return eKeyHandled;
      }
      break;

    case KEY_UP:
      if (m_submenus.size() > 1) {
        const int start_select = m_selected;
        while (--m_selected != start_select) {
          if (m_selected < static_cast<int>(0))
            m_selected = num_submenus - 1;
          if (m_submenus[m_selected]->GetType() == Type::Separator)
            continue;
          else
            break;
        }
        return eKeyHandled;
      }
      break;

    case KEY_RETURN:
      if (static_cast<size_t>(selected_idx) < num_submenus) {
        if (submenus[selected_idx]->Action() == MenuActionResult::Quit)
          return eQuitApplication;
        window.GetParent()->RemoveSubWindow(&window);
        return eKeyHandled;
      }
      break;

    case KEY_ESCAPE: // Beware: pressing escape key has 1 to 2 second delay in
                     // case other chars are entered for escaped sequences
      window.GetParent()->RemoveSubWindow(&window);
      return eKeyHandled;

    default:
      for (size_t i = 0; i < num_submenus; ++i) {
        Menu *menu = submenus[i].get();
        if (menu->GetKeyValue() == key) {
          SetSelectedSubmenuIndex(i);
          window.GetParent()->RemoveSubWindow(&window);
          if (menu->Action() == MenuActionResult::Quit)
            return eQuitApplication;
          return eKeyHandled;
        }
      }
      break;
    }
  } else if (menu_type == Menu::Type::Separator) {
  }
  return result;
}

class Application {
public:
  Application(FILE *in, FILE *out)
      : m_window_sp(), m_screen(nullptr), m_in(in), m_out(out) {}

  ~Application() {
    m_window_delegates.clear();
    m_window_sp.reset();
    if (m_screen) {
      ::delscreen(m_screen);
      m_screen = nullptr;
    }
  }

  void Initialize() {
    ::setlocale(LC_ALL, "");
    ::setlocale(LC_CTYPE, "");
    m_screen = ::newterm(nullptr, m_out, m_in);
    ::start_color();
    ::curs_set(0);
    ::noecho();
    ::keypad(stdscr, TRUE);
  }

  void Terminate() { ::endwin(); }

  void Run(Debugger &debugger) {
    bool done = false;
    int delay_in_tenths_of_a_second = 1;

    // Alas the threading model in curses is a bit lame so we need to resort to
    // polling every 0.5 seconds. We could poll for stdin ourselves and then
    // pass the keys down but then we need to translate all of the escape
    // sequences ourselves. So we resort to polling for input because we need
    // to receive async process events while in this loop.

    halfdelay(delay_in_tenths_of_a_second); // Poll using some number of tenths
                                            // of seconds seconds when calling
                                            // Window::GetChar()

    ListenerSP listener_sp(
        Listener::MakeListener("lldb.IOHandler.curses.Application"));
    ConstString broadcaster_class_process(Process::GetStaticBroadcasterClass());
    debugger.EnableForwardEvents(listener_sp);

    m_update_screen = true;
#if defined(__APPLE__)
    std::deque<int> escape_chars;
#endif

    while (!done) {
      if (m_update_screen) {
        m_window_sp->Draw(false);
        // All windows should be calling Window::DeferredRefresh() instead of
        // Window::Refresh() so we can do a single update and avoid any screen
        // blinking
        update_panels();

        // Cursor hiding isn't working on MacOSX, so hide it in the top left
        // corner
        m_window_sp->MoveCursor(0, 0);

        doupdate();
        m_update_screen = false;
      }

#if defined(__APPLE__)
      // Terminal.app doesn't map its function keys correctly, F1-F4 default
      // to: \033OP, \033OQ, \033OR, \033OS, so lets take care of this here if
      // possible
      int ch;
      if (escape_chars.empty())
        ch = m_window_sp->GetChar();
      else {
        ch = escape_chars.front();
        escape_chars.pop_front();
      }
      if (ch == KEY_ESCAPE) {
        int ch2 = m_window_sp->GetChar();
        if (ch2 == 'O') {
          int ch3 = m_window_sp->GetChar();
          switch (ch3) {
          case 'P':
            ch = KEY_F(1);
            break;
          case 'Q':
            ch = KEY_F(2);
            break;
          case 'R':
            ch = KEY_F(3);
            break;
          case 'S':
            ch = KEY_F(4);
            break;
          default:
            escape_chars.push_back(ch2);
            if (ch3 != -1)
              escape_chars.push_back(ch3);
            break;
          }
        } else if (ch2 != -1)
          escape_chars.push_back(ch2);
      }
#else
      int ch = m_window_sp->GetChar();

#endif
      if (ch == -1) {
        if (feof(m_in) || ferror(m_in)) {
          done = true;
        } else {
          // Just a timeout from using halfdelay(), check for events
          EventSP event_sp;
          while (listener_sp->PeekAtNextEvent()) {
            listener_sp->GetEvent(event_sp, std::chrono::seconds(0));

            if (event_sp) {
              Broadcaster *broadcaster = event_sp->GetBroadcaster();
              if (broadcaster) {
                // uint32_t event_type = event_sp->GetType();
                ConstString broadcaster_class(
                    broadcaster->GetBroadcasterClass());
                if (broadcaster_class == broadcaster_class_process) {
                  debugger.GetCommandInterpreter().UpdateExecutionContext(
                      nullptr);
                  m_update_screen = true;
                  continue; // Don't get any key, just update our view
                }
              }
            }
          }
        }
      } else {
        HandleCharResult key_result = m_window_sp->HandleChar(ch);
        switch (key_result) {
        case eKeyHandled:
          debugger.GetCommandInterpreter().UpdateExecutionContext(nullptr);
          m_update_screen = true;
          break;
        case eKeyNotHandled:
          if (ch == 12) { // Ctrl+L, force full redraw
            redrawwin(m_window_sp->get());
            m_update_screen = true;
          }
          break;
        case eQuitApplication:
          done = true;
          break;
        }
      }
    }

    debugger.CancelForwardEvents(listener_sp);
  }

  WindowSP &GetMainWindow() {
    if (!m_window_sp)
      m_window_sp = std::make_shared<Window>("main", stdscr, false);
    return m_window_sp;
  }

  void TerminalSizeChanged() {
    ::endwin();
    ::refresh();
    Rect content_bounds = m_window_sp->GetFrame();
    m_window_sp->SetBounds(content_bounds);
    if (WindowSP menubar_window_sp = m_window_sp->FindSubWindow("Menubar"))
      menubar_window_sp->SetBounds(content_bounds.MakeMenuBar());
    if (WindowSP status_window_sp = m_window_sp->FindSubWindow("Status"))
      status_window_sp->SetBounds(content_bounds.MakeStatusBar());

    WindowSP source_window_sp = m_window_sp->FindSubWindow("Source");
    WindowSP variables_window_sp = m_window_sp->FindSubWindow("Variables");
    WindowSP registers_window_sp = m_window_sp->FindSubWindow("Registers");
    WindowSP threads_window_sp = m_window_sp->FindSubWindow("Threads");

    Rect threads_bounds;
    Rect source_variables_bounds;
    content_bounds.VerticalSplitPercentage(0.80, source_variables_bounds,
                                           threads_bounds);
    if (threads_window_sp)
      threads_window_sp->SetBounds(threads_bounds);
    else
      source_variables_bounds = content_bounds;

    Rect source_bounds;
    Rect variables_registers_bounds;
    source_variables_bounds.HorizontalSplitPercentage(
        0.70, source_bounds, variables_registers_bounds);
    if (variables_window_sp || registers_window_sp) {
      if (variables_window_sp && registers_window_sp) {
        Rect variables_bounds;
        Rect registers_bounds;
        variables_registers_bounds.VerticalSplitPercentage(
            0.50, variables_bounds, registers_bounds);
        variables_window_sp->SetBounds(variables_bounds);
        registers_window_sp->SetBounds(registers_bounds);
      } else if (variables_window_sp) {
        variables_window_sp->SetBounds(variables_registers_bounds);
      } else {
        registers_window_sp->SetBounds(variables_registers_bounds);
      }
    } else {
      source_bounds = source_variables_bounds;
    }

    source_window_sp->SetBounds(source_bounds);

    touchwin(stdscr);
    redrawwin(m_window_sp->get());
    m_update_screen = true;
  }

protected:
  WindowSP m_window_sp;
  WindowDelegates m_window_delegates;
  SCREEN *m_screen;
  FILE *m_in;
  FILE *m_out;
  bool m_update_screen = false;
};

} // namespace curses

using namespace curses;

struct Row {
  ValueObjectManager value;
  Row *parent;
  // The process stop ID when the children were calculated.
  uint32_t children_stop_id;
  int row_idx;
  int x;
  int y;
  bool might_have_children;
  bool expanded;
  bool calculated_children;
  std::vector<Row> children;

  Row(const ValueObjectSP &v, Row *p)
      : value(v, lldb::eDynamicDontRunTarget, true), parent(p), row_idx(0),
        x(1), y(1), might_have_children(v ? v->MightHaveChildren() : false),
        expanded(false), calculated_children(false), children() {}

  size_t GetDepth() const {
    if (parent)
      return 1 + parent->GetDepth();
    return 0;
  }

  void Expand() { expanded = true; }

  std::vector<Row> &GetChildren() {
    ProcessSP process_sp = value.GetProcessSP();
    auto stop_id = process_sp->GetStopID();
    if (process_sp && stop_id != children_stop_id) {
      children_stop_id = stop_id;
      calculated_children = false;
    }
    if (!calculated_children) {
      children.clear();
      calculated_children = true;
      ValueObjectSP valobj = value.GetSP();
      if (valobj) {
        const size_t num_children = valobj->GetNumChildren();
        for (size_t i = 0; i < num_children; ++i) {
          children.push_back(Row(valobj->GetChildAtIndex(i, true), this));
        }
      }
    }
    return children;
  }

  void Unexpand() {
    expanded = false;
    calculated_children = false;
    children.clear();
  }

  void DrawTree(Window &window) {
    if (parent)
      parent->DrawTreeForChild(window, this, 0);

    if (might_have_children) {
      // It we can get UTF8 characters to work we should try to use the
      // "symbol" UTF8 string below
      //            const char *symbol = "";
      //            if (row.expanded)
      //                symbol = "\xe2\x96\xbd ";
      //            else
      //                symbol = "\xe2\x96\xb7 ";
      //            window.PutCString (symbol);

      // The ACS_DARROW and ACS_RARROW don't look very nice they are just a 'v'
      // or '>' character...
      //            if (expanded)
      //                window.PutChar (ACS_DARROW);
      //            else
      //                window.PutChar (ACS_RARROW);
      // Since we can't find any good looking right arrow/down arrow symbols,
      // just use a diamond...
      window.PutChar(ACS_DIAMOND);
      window.PutChar(ACS_HLINE);
    }
  }

  void DrawTreeForChild(Window &window, Row *child, uint32_t reverse_depth) {
    if (parent)
      parent->DrawTreeForChild(window, this, reverse_depth + 1);

    if (&GetChildren().back() == child) {
      // Last child
      if (reverse_depth == 0) {
        window.PutChar(ACS_LLCORNER);
        window.PutChar(ACS_HLINE);
      } else {
        window.PutChar(' ');
        window.PutChar(' ');
      }
    } else {
      if (reverse_depth == 0) {
        window.PutChar(ACS_LTEE);
        window.PutChar(ACS_HLINE);
      } else {
        window.PutChar(ACS_VLINE);
        window.PutChar(' ');
      }
    }
  }
};

struct DisplayOptions {
  bool show_types;
};

class TreeItem;

class TreeDelegate {
public:
  TreeDelegate() = default;
  virtual ~TreeDelegate() = default;

  virtual void TreeDelegateDrawTreeItem(TreeItem &item, Window &window) = 0;
  virtual void TreeDelegateGenerateChildren(TreeItem &item) = 0;
  virtual bool TreeDelegateItemSelected(
      TreeItem &item) = 0; // Return true if we need to update views
};

typedef std::shared_ptr<TreeDelegate> TreeDelegateSP;

class TreeItem {
public:
  TreeItem(TreeItem *parent, TreeDelegate &delegate, bool might_have_children)
      : m_parent(parent), m_delegate(delegate), m_user_data(nullptr),
        m_identifier(0), m_row_idx(-1), m_children(),
        m_might_have_children(might_have_children), m_is_expanded(false) {}

  TreeItem &operator=(const TreeItem &rhs) {
    if (this != &rhs) {
      m_parent = rhs.m_parent;
      m_delegate = rhs.m_delegate;
      m_user_data = rhs.m_user_data;
      m_identifier = rhs.m_identifier;
      m_row_idx = rhs.m_row_idx;
      m_children = rhs.m_children;
      m_might_have_children = rhs.m_might_have_children;
      m_is_expanded = rhs.m_is_expanded;
    }
    return *this;
  }

  TreeItem(const TreeItem &) = default;

  size_t GetDepth() const {
    if (m_parent)
      return 1 + m_parent->GetDepth();
    return 0;
  }

  int GetRowIndex() const { return m_row_idx; }

  void ClearChildren() { m_children.clear(); }

  void Resize(size_t n, const TreeItem &t) { m_children.resize(n, t); }

  TreeItem &operator[](size_t i) { return m_children[i]; }

  void SetRowIndex(int row_idx) { m_row_idx = row_idx; }

  size_t GetNumChildren() {
    m_delegate.TreeDelegateGenerateChildren(*this);
    return m_children.size();
  }

  void ItemWasSelected() { m_delegate.TreeDelegateItemSelected(*this); }

  void CalculateRowIndexes(int &row_idx) {
    SetRowIndex(row_idx);
    ++row_idx;

    const bool expanded = IsExpanded();

    // The root item must calculate its children, or we must calculate the
    // number of children if the item is expanded
    if (m_parent == nullptr || expanded)
      GetNumChildren();

    for (auto &item : m_children) {
      if (expanded)
        item.CalculateRowIndexes(row_idx);
      else
        item.SetRowIndex(-1);
    }
  }

  TreeItem *GetParent() { return m_parent; }

  bool IsExpanded() const { return m_is_expanded; }

  void Expand() { m_is_expanded = true; }

  void Unexpand() { m_is_expanded = false; }

  bool Draw(Window &window, const int first_visible_row,
            const uint32_t selected_row_idx, int &row_idx, int &num_rows_left) {
    if (num_rows_left <= 0)
      return false;

    if (m_row_idx >= first_visible_row) {
      window.MoveCursor(2, row_idx + 1);

      if (m_parent)
        m_parent->DrawTreeForChild(window, this, 0);

      if (m_might_have_children) {
        // It we can get UTF8 characters to work we should try to use the
        // "symbol" UTF8 string below
        //            const char *symbol = "";
        //            if (row.expanded)
        //                symbol = "\xe2\x96\xbd ";
        //            else
        //                symbol = "\xe2\x96\xb7 ";
        //            window.PutCString (symbol);

        // The ACS_DARROW and ACS_RARROW don't look very nice they are just a
        // 'v' or '>' character...
        //            if (expanded)
        //                window.PutChar (ACS_DARROW);
        //            else
        //                window.PutChar (ACS_RARROW);
        // Since we can't find any good looking right arrow/down arrow symbols,
        // just use a diamond...
        window.PutChar(ACS_DIAMOND);
        window.PutChar(ACS_HLINE);
      }
      bool highlight = (selected_row_idx == static_cast<size_t>(m_row_idx)) &&
                       window.IsActive();

      if (highlight)
        window.AttributeOn(A_REVERSE);

      m_delegate.TreeDelegateDrawTreeItem(*this, window);

      if (highlight)
        window.AttributeOff(A_REVERSE);
      ++row_idx;
      --num_rows_left;
    }

    if (num_rows_left <= 0)
      return false; // We are done drawing...

    if (IsExpanded()) {
      for (auto &item : m_children) {
        // If we displayed all the rows and item.Draw() returns false we are
        // done drawing and can exit this for loop
        if (!item.Draw(window, first_visible_row, selected_row_idx, row_idx,
                       num_rows_left))
          break;
      }
    }
    return num_rows_left >= 0; // Return true if not done drawing yet
  }

  void DrawTreeForChild(Window &window, TreeItem *child,
                        uint32_t reverse_depth) {
    if (m_parent)
      m_parent->DrawTreeForChild(window, this, reverse_depth + 1);

    if (&m_children.back() == child) {
      // Last child
      if (reverse_depth == 0) {
        window.PutChar(ACS_LLCORNER);
        window.PutChar(ACS_HLINE);
      } else {
        window.PutChar(' ');
        window.PutChar(' ');
      }
    } else {
      if (reverse_depth == 0) {
        window.PutChar(ACS_LTEE);
        window.PutChar(ACS_HLINE);
      } else {
        window.PutChar(ACS_VLINE);
        window.PutChar(' ');
      }
    }
  }

  TreeItem *GetItemForRowIndex(uint32_t row_idx) {
    if (static_cast<uint32_t>(m_row_idx) == row_idx)
      return this;
    if (m_children.empty())
      return nullptr;
    if (IsExpanded()) {
      for (auto &item : m_children) {
        TreeItem *selected_item_ptr = item.GetItemForRowIndex(row_idx);
        if (selected_item_ptr)
          return selected_item_ptr;
      }
    }
    return nullptr;
  }

  void *GetUserData() const { return m_user_data; }

  void SetUserData(void *user_data) { m_user_data = user_data; }

  uint64_t GetIdentifier() const { return m_identifier; }

  void SetIdentifier(uint64_t identifier) { m_identifier = identifier; }

  void SetMightHaveChildren(bool b) { m_might_have_children = b; }

protected:
  TreeItem *m_parent;
  TreeDelegate &m_delegate;
  void *m_user_data;
  uint64_t m_identifier;
  int m_row_idx; // Zero based visible row index, -1 if not visible or for the
                 // root item
  std::vector<TreeItem> m_children;
  bool m_might_have_children;
  bool m_is_expanded;
};

class TreeWindowDelegate : public WindowDelegate {
public:
  TreeWindowDelegate(Debugger &debugger, const TreeDelegateSP &delegate_sp)
      : m_debugger(debugger), m_delegate_sp(delegate_sp),
        m_root(nullptr, *delegate_sp, true), m_selected_item(nullptr),
        m_num_rows(0), m_selected_row_idx(0), m_first_visible_row(0),
        m_min_x(0), m_min_y(0), m_max_x(0), m_max_y(0) {}

  int NumVisibleRows() const { return m_max_y - m_min_y; }

  bool WindowDelegateDraw(Window &window, bool force) override {
    ExecutionContext exe_ctx(
        m_debugger.GetCommandInterpreter().GetExecutionContext());
    Process *process = exe_ctx.GetProcessPtr();

    bool display_content = false;
    if (process) {
      StateType state = process->GetState();
      if (StateIsStoppedState(state, true)) {
        // We are stopped, so it is ok to
        display_content = true;
      } else if (StateIsRunningState(state)) {
        return true; // Don't do any updating when we are running
      }
    }

    m_min_x = 2;
    m_min_y = 1;
    m_max_x = window.GetWidth() - 1;
    m_max_y = window.GetHeight() - 1;

    window.Erase();
    window.DrawTitleBox(window.GetName());

    if (display_content) {
      const int num_visible_rows = NumVisibleRows();
      m_num_rows = 0;
      m_root.CalculateRowIndexes(m_num_rows);

      // If we unexpanded while having something selected our total number of
      // rows is less than the num visible rows, then make sure we show all the
      // rows by setting the first visible row accordingly.
      if (m_first_visible_row > 0 && m_num_rows < num_visible_rows)
        m_first_visible_row = 0;

      // Make sure the selected row is always visible
      if (m_selected_row_idx < m_first_visible_row)
        m_first_visible_row = m_selected_row_idx;
      else if (m_first_visible_row + num_visible_rows <= m_selected_row_idx)
        m_first_visible_row = m_selected_row_idx - num_visible_rows + 1;

      int row_idx = 0;
      int num_rows_left = num_visible_rows;
      m_root.Draw(window, m_first_visible_row, m_selected_row_idx, row_idx,
                  num_rows_left);
      // Get the selected row
      m_selected_item = m_root.GetItemForRowIndex(m_selected_row_idx);
    } else {
      m_selected_item = nullptr;
    }

    return true; // Drawing handled
  }

  const char *WindowDelegateGetHelpText() override {
    return "Thread window keyboard shortcuts:";
  }

  KeyHelp *WindowDelegateGetKeyHelp() override {
    static curses::KeyHelp g_source_view_key_help[] = {
        {KEY_UP, "Select previous item"},
        {KEY_DOWN, "Select next item"},
        {KEY_RIGHT, "Expand the selected item"},
        {KEY_LEFT,
         "Unexpand the selected item or select parent if not expanded"},
        {KEY_PPAGE, "Page up"},
        {KEY_NPAGE, "Page down"},
        {'h', "Show help dialog"},
        {' ', "Toggle item expansion"},
        {',', "Page up"},
        {'.', "Page down"},
        {'\0', nullptr}};
    return g_source_view_key_help;
  }

  HandleCharResult WindowDelegateHandleChar(Window &window, int c) override {
    switch (c) {
    case ',':
    case KEY_PPAGE:
      // Page up key
      if (m_first_visible_row > 0) {
        if (m_first_visible_row > m_max_y)
          m_first_visible_row -= m_max_y;
        else
          m_first_visible_row = 0;
        m_selected_row_idx = m_first_visible_row;
        m_selected_item = m_root.GetItemForRowIndex(m_selected_row_idx);
        if (m_selected_item)
          m_selected_item->ItemWasSelected();
      }
      return eKeyHandled;

    case '.':
    case KEY_NPAGE:
      // Page down key
      if (m_num_rows > m_max_y) {
        if (m_first_visible_row + m_max_y < m_num_rows) {
          m_first_visible_row += m_max_y;
          m_selected_row_idx = m_first_visible_row;
          m_selected_item = m_root.GetItemForRowIndex(m_selected_row_idx);
          if (m_selected_item)
            m_selected_item->ItemWasSelected();
        }
      }
      return eKeyHandled;

    case KEY_UP:
      if (m_selected_row_idx > 0) {
        --m_selected_row_idx;
        m_selected_item = m_root.GetItemForRowIndex(m_selected_row_idx);
        if (m_selected_item)
          m_selected_item->ItemWasSelected();
      }
      return eKeyHandled;

    case KEY_DOWN:
      if (m_selected_row_idx + 1 < m_num_rows) {
        ++m_selected_row_idx;
        m_selected_item = m_root.GetItemForRowIndex(m_selected_row_idx);
        if (m_selected_item)
          m_selected_item->ItemWasSelected();
      }
      return eKeyHandled;

    case KEY_RIGHT:
      if (m_selected_item) {
        if (!m_selected_item->IsExpanded())
          m_selected_item->Expand();
      }
      return eKeyHandled;

    case KEY_LEFT:
      if (m_selected_item) {
        if (m_selected_item->IsExpanded())
          m_selected_item->Unexpand();
        else if (m_selected_item->GetParent()) {
          m_selected_row_idx = m_selected_item->GetParent()->GetRowIndex();
          m_selected_item = m_root.GetItemForRowIndex(m_selected_row_idx);
          if (m_selected_item)
            m_selected_item->ItemWasSelected();
        }
      }
      return eKeyHandled;

    case ' ':
      // Toggle expansion state when SPACE is pressed
      if (m_selected_item) {
        if (m_selected_item->IsExpanded())
          m_selected_item->Unexpand();
        else
          m_selected_item->Expand();
      }
      return eKeyHandled;

    case 'h':
      window.CreateHelpSubwindow();
      return eKeyHandled;

    default:
      break;
    }
    return eKeyNotHandled;
  }

protected:
  Debugger &m_debugger;
  TreeDelegateSP m_delegate_sp;
  TreeItem m_root;
  TreeItem *m_selected_item;
  int m_num_rows;
  int m_selected_row_idx;
  int m_first_visible_row;
  int m_min_x;
  int m_min_y;
  int m_max_x;
  int m_max_y;
};

class FrameTreeDelegate : public TreeDelegate {
public:
  FrameTreeDelegate() : TreeDelegate() {
    FormatEntity::Parse(
        "frame #${frame.index}: {${function.name}${function.pc-offset}}}",
        m_format);
  }

  ~FrameTreeDelegate() override = default;

  void TreeDelegateDrawTreeItem(TreeItem &item, Window &window) override {
    Thread *thread = (Thread *)item.GetUserData();
    if (thread) {
      const uint64_t frame_idx = item.GetIdentifier();
      StackFrameSP frame_sp = thread->GetStackFrameAtIndex(frame_idx);
      if (frame_sp) {
        StreamString strm;
        const SymbolContext &sc =
            frame_sp->GetSymbolContext(eSymbolContextEverything);
        ExecutionContext exe_ctx(frame_sp);
        if (FormatEntity::Format(m_format, strm, &sc, &exe_ctx, nullptr,
                                 nullptr, false, false)) {
          int right_pad = 1;
          window.PutCStringTruncated(right_pad, strm.GetString().str().c_str());
        }
      }
    }
  }

  void TreeDelegateGenerateChildren(TreeItem &item) override {
    // No children for frames yet...
  }

  bool TreeDelegateItemSelected(TreeItem &item) override {
    Thread *thread = (Thread *)item.GetUserData();
    if (thread) {
      thread->GetProcess()->GetThreadList().SetSelectedThreadByID(
          thread->GetID());
      const uint64_t frame_idx = item.GetIdentifier();
      thread->SetSelectedFrameByIndex(frame_idx);
      return true;
    }
    return false;
  }

protected:
  FormatEntity::Entry m_format;
};

class ThreadTreeDelegate : public TreeDelegate {
public:
  ThreadTreeDelegate(Debugger &debugger)
      : TreeDelegate(), m_debugger(debugger), m_tid(LLDB_INVALID_THREAD_ID),
        m_stop_id(UINT32_MAX) {
    FormatEntity::Parse("thread #${thread.index}: tid = ${thread.id}{, stop "
                        "reason = ${thread.stop-reason}}",
                        m_format);
  }

  ~ThreadTreeDelegate() override = default;

  ProcessSP GetProcess() {
    return m_debugger.GetCommandInterpreter()
        .GetExecutionContext()
        .GetProcessSP();
  }

  ThreadSP GetThread(const TreeItem &item) {
    ProcessSP process_sp = GetProcess();
    if (process_sp)
      return process_sp->GetThreadList().FindThreadByID(item.GetIdentifier());
    return ThreadSP();
  }

  void TreeDelegateDrawTreeItem(TreeItem &item, Window &window) override {
    ThreadSP thread_sp = GetThread(item);
    if (thread_sp) {
      StreamString strm;
      ExecutionContext exe_ctx(thread_sp);
      if (FormatEntity::Format(m_format, strm, nullptr, &exe_ctx, nullptr,
                               nullptr, false, false)) {
        int right_pad = 1;
        window.PutCStringTruncated(right_pad, strm.GetString().str().c_str());
      }
    }
  }

  void TreeDelegateGenerateChildren(TreeItem &item) override {
    ProcessSP process_sp = GetProcess();
    if (process_sp && process_sp->IsAlive()) {
      StateType state = process_sp->GetState();
      if (StateIsStoppedState(state, true)) {
        ThreadSP thread_sp = GetThread(item);
        if (thread_sp) {
          if (m_stop_id == process_sp->GetStopID() &&
              thread_sp->GetID() == m_tid)
            return; // Children are already up to date
          if (!m_frame_delegate_sp) {
            // Always expand the thread item the first time we show it
            m_frame_delegate_sp = std::make_shared<FrameTreeDelegate>();
          }

          m_stop_id = process_sp->GetStopID();
          m_tid = thread_sp->GetID();

          TreeItem t(&item, *m_frame_delegate_sp, false);
          size_t num_frames = thread_sp->GetStackFrameCount();
          item.Resize(num_frames, t);
          for (size_t i = 0; i < num_frames; ++i) {
            item[i].SetUserData(thread_sp.get());
            item[i].SetIdentifier(i);
          }
        }
        return;
      }
    }
    item.ClearChildren();
  }

  bool TreeDelegateItemSelected(TreeItem &item) override {
    ProcessSP process_sp = GetProcess();
    if (process_sp && process_sp->IsAlive()) {
      StateType state = process_sp->GetState();
      if (StateIsStoppedState(state, true)) {
        ThreadSP thread_sp = GetThread(item);
        if (thread_sp) {
          ThreadList &thread_list = thread_sp->GetProcess()->GetThreadList();
          std::lock_guard<std::recursive_mutex> guard(thread_list.GetMutex());
          ThreadSP selected_thread_sp = thread_list.GetSelectedThread();
          if (selected_thread_sp->GetID() != thread_sp->GetID()) {
            thread_list.SetSelectedThreadByID(thread_sp->GetID());
            return true;
          }
        }
      }
    }
    return false;
  }

protected:
  Debugger &m_debugger;
  std::shared_ptr<FrameTreeDelegate> m_frame_delegate_sp;
  lldb::user_id_t m_tid;
  uint32_t m_stop_id;
  FormatEntity::Entry m_format;
};

class ThreadsTreeDelegate : public TreeDelegate {
public:
  ThreadsTreeDelegate(Debugger &debugger)
      : TreeDelegate(), m_thread_delegate_sp(), m_debugger(debugger),
        m_stop_id(UINT32_MAX) {
    FormatEntity::Parse("process ${process.id}{, name = ${process.name}}",
                        m_format);
  }

  ~ThreadsTreeDelegate() override = default;

  ProcessSP GetProcess() {
    return m_debugger.GetCommandInterpreter()
        .GetExecutionContext()
        .GetProcessSP();
  }

  void TreeDelegateDrawTreeItem(TreeItem &item, Window &window) override {
    ProcessSP process_sp = GetProcess();
    if (process_sp && process_sp->IsAlive()) {
      StreamString strm;
      ExecutionContext exe_ctx(process_sp);
      if (FormatEntity::Format(m_format, strm, nullptr, &exe_ctx, nullptr,
                               nullptr, false, false)) {
        int right_pad = 1;
        window.PutCStringTruncated(right_pad, strm.GetString().str().c_str());
      }
    }
  }

  void TreeDelegateGenerateChildren(TreeItem &item) override {
    ProcessSP process_sp = GetProcess();
    if (process_sp && process_sp->IsAlive()) {
      StateType state = process_sp->GetState();
      if (StateIsStoppedState(state, true)) {
        const uint32_t stop_id = process_sp->GetStopID();
        if (m_stop_id == stop_id)
          return; // Children are already up to date

        m_stop_id = stop_id;

        if (!m_thread_delegate_sp) {
          // Always expand the thread item the first time we show it
          // item.Expand();
          m_thread_delegate_sp =
              std::make_shared<ThreadTreeDelegate>(m_debugger);
        }

        TreeItem t(&item, *m_thread_delegate_sp, false);
        ThreadList &threads = process_sp->GetThreadList();
        std::lock_guard<std::recursive_mutex> guard(threads.GetMutex());
        size_t num_threads = threads.GetSize();
        item.Resize(num_threads, t);
        for (size_t i = 0; i < num_threads; ++i) {
          item[i].SetIdentifier(threads.GetThreadAtIndex(i)->GetID());
          item[i].SetMightHaveChildren(true);
        }
        return;
      }
    }
    item.ClearChildren();
  }

  bool TreeDelegateItemSelected(TreeItem &item) override { return false; }

protected:
  std::shared_ptr<ThreadTreeDelegate> m_thread_delegate_sp;
  Debugger &m_debugger;
  uint32_t m_stop_id;
  FormatEntity::Entry m_format;
};

class ValueObjectListDelegate : public WindowDelegate {
public:
  ValueObjectListDelegate()
      : m_rows(), m_selected_row(nullptr), m_selected_row_idx(0),
        m_first_visible_row(0), m_num_rows(0), m_max_x(0), m_max_y(0) {}

  ValueObjectListDelegate(ValueObjectList &valobj_list)
      : m_rows(), m_selected_row(nullptr), m_selected_row_idx(0),
        m_first_visible_row(0), m_num_rows(0), m_max_x(0), m_max_y(0) {
    SetValues(valobj_list);
  }

  ~ValueObjectListDelegate() override = default;

  void SetValues(ValueObjectList &valobj_list) {
    m_selected_row = nullptr;
    m_selected_row_idx = 0;
    m_first_visible_row = 0;
    m_num_rows = 0;
    m_rows.clear();
    for (auto &valobj_sp : valobj_list.GetObjects())
      m_rows.push_back(Row(valobj_sp, nullptr));
  }

  bool WindowDelegateDraw(Window &window, bool force) override {
    m_num_rows = 0;
    m_min_x = 2;
    m_min_y = 1;
    m_max_x = window.GetWidth() - 1;
    m_max_y = window.GetHeight() - 1;

    window.Erase();
    window.DrawTitleBox(window.GetName());

    const int num_visible_rows = NumVisibleRows();
    const int num_rows = CalculateTotalNumberRows(m_rows);

    // If we unexpanded while having something selected our total number of
    // rows is less than the num visible rows, then make sure we show all the
    // rows by setting the first visible row accordingly.
    if (m_first_visible_row > 0 && num_rows < num_visible_rows)
      m_first_visible_row = 0;

    // Make sure the selected row is always visible
    if (m_selected_row_idx < m_first_visible_row)
      m_first_visible_row = m_selected_row_idx;
    else if (m_first_visible_row + num_visible_rows <= m_selected_row_idx)
      m_first_visible_row = m_selected_row_idx - num_visible_rows + 1;

    DisplayRows(window, m_rows, g_options);

    // Get the selected row
    m_selected_row = GetRowForRowIndex(m_selected_row_idx);
    // Keep the cursor on the selected row so the highlight and the cursor are
    // always on the same line
    if (m_selected_row)
      window.MoveCursor(m_selected_row->x, m_selected_row->y);

    return true; // Drawing handled
  }

  KeyHelp *WindowDelegateGetKeyHelp() override {
    static curses::KeyHelp g_source_view_key_help[] = {
        {KEY_UP, "Select previous item"},
        {KEY_DOWN, "Select next item"},
        {KEY_RIGHT, "Expand selected item"},
        {KEY_LEFT, "Unexpand selected item or select parent if not expanded"},
        {KEY_PPAGE, "Page up"},
        {KEY_NPAGE, "Page down"},
        {'A', "Format as annotated address"},
        {'b', "Format as binary"},
        {'B', "Format as hex bytes with ASCII"},
        {'c', "Format as character"},
        {'d', "Format as a signed integer"},
        {'D', "Format selected value using the default format for the type"},
        {'f', "Format as float"},
        {'h', "Show help dialog"},
        {'i', "Format as instructions"},
        {'o', "Format as octal"},
        {'p', "Format as pointer"},
        {'s', "Format as C string"},
        {'t', "Toggle showing/hiding type names"},
        {'u', "Format as an unsigned integer"},
        {'x', "Format as hex"},
        {'X', "Format as uppercase hex"},
        {' ', "Toggle item expansion"},
        {',', "Page up"},
        {'.', "Page down"},
        {'\0', nullptr}};
    return g_source_view_key_help;
  }

  HandleCharResult WindowDelegateHandleChar(Window &window, int c) override {
    switch (c) {
    case 'x':
    case 'X':
    case 'o':
    case 's':
    case 'u':
    case 'd':
    case 'D':
    case 'i':
    case 'A':
    case 'p':
    case 'c':
    case 'b':
    case 'B':
    case 'f':
      // Change the format for the currently selected item
      if (m_selected_row) {
        auto valobj_sp = m_selected_row->value.GetSP();
        if (valobj_sp)
          valobj_sp->SetFormat(FormatForChar(c));
      }
      return eKeyHandled;

    case 't':
      // Toggle showing type names
      g_options.show_types = !g_options.show_types;
      return eKeyHandled;

    case ',':
    case KEY_PPAGE:
      // Page up key
      if (m_first_visible_row > 0) {
        if (static_cast<int>(m_first_visible_row) > m_max_y)
          m_first_visible_row -= m_max_y;
        else
          m_first_visible_row = 0;
        m_selected_row_idx = m_first_visible_row;
      }
      return eKeyHandled;

    case '.':
    case KEY_NPAGE:
      // Page down key
      if (m_num_rows > static_cast<size_t>(m_max_y)) {
        if (m_first_visible_row + m_max_y < m_num_rows) {
          m_first_visible_row += m_max_y;
          m_selected_row_idx = m_first_visible_row;
        }
      }
      return eKeyHandled;

    case KEY_UP:
      if (m_selected_row_idx > 0)
        --m_selected_row_idx;
      return eKeyHandled;

    case KEY_DOWN:
      if (m_selected_row_idx + 1 < m_num_rows)
        ++m_selected_row_idx;
      return eKeyHandled;

    case KEY_RIGHT:
      if (m_selected_row) {
        if (!m_selected_row->expanded)
          m_selected_row->Expand();
      }
      return eKeyHandled;

    case KEY_LEFT:
      if (m_selected_row) {
        if (m_selected_row->expanded)
          m_selected_row->Unexpand();
        else if (m_selected_row->parent)
          m_selected_row_idx = m_selected_row->parent->row_idx;
      }
      return eKeyHandled;

    case ' ':
      // Toggle expansion state when SPACE is pressed
      if (m_selected_row) {
        if (m_selected_row->expanded)
          m_selected_row->Unexpand();
        else
          m_selected_row->Expand();
      }
      return eKeyHandled;

    case 'h':
      window.CreateHelpSubwindow();
      return eKeyHandled;

    default:
      break;
    }
    return eKeyNotHandled;
  }

protected:
  std::vector<Row> m_rows;
  Row *m_selected_row;
  uint32_t m_selected_row_idx;
  uint32_t m_first_visible_row;
  uint32_t m_num_rows;
  int m_min_x;
  int m_min_y;
  int m_max_x;
  int m_max_y;

  static Format FormatForChar(int c) {
    switch (c) {
    case 'x':
      return eFormatHex;
    case 'X':
      return eFormatHexUppercase;
    case 'o':
      return eFormatOctal;
    case 's':
      return eFormatCString;
    case 'u':
      return eFormatUnsigned;
    case 'd':
      return eFormatDecimal;
    case 'D':
      return eFormatDefault;
    case 'i':
      return eFormatInstruction;
    case 'A':
      return eFormatAddressInfo;
    case 'p':
      return eFormatPointer;
    case 'c':
      return eFormatChar;
    case 'b':
      return eFormatBinary;
    case 'B':
      return eFormatBytesWithASCII;
    case 'f':
      return eFormatFloat;
    }
    return eFormatDefault;
  }

  bool DisplayRowObject(Window &window, Row &row, DisplayOptions &options,
                        bool highlight, bool last_child) {
    ValueObject *valobj = row.value.GetSP().get();

    if (valobj == nullptr)
      return false;

    const char *type_name =
        options.show_types ? valobj->GetTypeName().GetCString() : nullptr;
    const char *name = valobj->GetName().GetCString();
    const char *value = valobj->GetValueAsCString();
    const char *summary = valobj->GetSummaryAsCString();

    window.MoveCursor(row.x, row.y);

    row.DrawTree(window);

    if (highlight)
      window.AttributeOn(A_REVERSE);

    if (type_name && type_name[0])
      window.PrintfTruncated(1, "(%s) ", type_name);

    if (name && name[0])
      window.PutCStringTruncated(1, name);

    attr_t changd_attr = 0;
    if (valobj->GetValueDidChange())
      changd_attr = COLOR_PAIR(RedOnBlack) | A_BOLD;

    if (value && value[0]) {
      window.PutCStringTruncated(1, " = ");
      if (changd_attr)
        window.AttributeOn(changd_attr);
      window.PutCStringTruncated(1, value);
      if (changd_attr)
        window.AttributeOff(changd_attr);
    }

    if (summary && summary[0]) {
      window.PutCStringTruncated(1, " ");
      if (changd_attr)
        window.AttributeOn(changd_attr);
      window.PutCStringTruncated(1, summary);
      if (changd_attr)
        window.AttributeOff(changd_attr);
    }

    if (highlight)
      window.AttributeOff(A_REVERSE);

    return true;
  }

  void DisplayRows(Window &window, std::vector<Row> &rows,
                   DisplayOptions &options) {
    // >   0x25B7
    // \/  0x25BD

    bool window_is_active = window.IsActive();
    for (auto &row : rows) {
      const bool last_child = row.parent && &rows[rows.size() - 1] == &row;
      // Save the row index in each Row structure
      row.row_idx = m_num_rows;
      if ((m_num_rows >= m_first_visible_row) &&
          ((m_num_rows - m_first_visible_row) <
           static_cast<size_t>(NumVisibleRows()))) {
        row.x = m_min_x;
        row.y = m_num_rows - m_first_visible_row + 1;
        if (DisplayRowObject(window, row, options,
                             window_is_active &&
                                 m_num_rows == m_selected_row_idx,
                             last_child)) {
          ++m_num_rows;
        } else {
          row.x = 0;
          row.y = 0;
        }
      } else {
        row.x = 0;
        row.y = 0;
        ++m_num_rows;
      }

      auto &children = row.GetChildren();
      if (row.expanded && !children.empty()) {
        DisplayRows(window, children, options);
      }
    }
  }

  int CalculateTotalNumberRows(std::vector<Row> &rows) {
    int row_count = 0;
    for (auto &row : rows) {
      ++row_count;
      if (row.expanded)
        row_count += CalculateTotalNumberRows(row.GetChildren());
    }
    return row_count;
  }

  static Row *GetRowForRowIndexImpl(std::vector<Row> &rows, size_t &row_index) {
    for (auto &row : rows) {
      if (row_index == 0)
        return &row;
      else {
        --row_index;
        auto &children = row.GetChildren();
        if (row.expanded && !children.empty()) {
          Row *result = GetRowForRowIndexImpl(children, row_index);
          if (result)
            return result;
        }
      }
    }
    return nullptr;
  }

  Row *GetRowForRowIndex(size_t row_index) {
    return GetRowForRowIndexImpl(m_rows, row_index);
  }

  int NumVisibleRows() const { return m_max_y - m_min_y; }

  static DisplayOptions g_options;
};

class FrameVariablesWindowDelegate : public ValueObjectListDelegate {
public:
  FrameVariablesWindowDelegate(Debugger &debugger)
      : ValueObjectListDelegate(), m_debugger(debugger),
        m_frame_block(nullptr) {}

  ~FrameVariablesWindowDelegate() override = default;

  const char *WindowDelegateGetHelpText() override {
    return "Frame variable window keyboard shortcuts:";
  }

  bool WindowDelegateDraw(Window &window, bool force) override {
    ExecutionContext exe_ctx(
        m_debugger.GetCommandInterpreter().GetExecutionContext());
    Process *process = exe_ctx.GetProcessPtr();
    Block *frame_block = nullptr;
    StackFrame *frame = nullptr;

    if (process) {
      StateType state = process->GetState();
      if (StateIsStoppedState(state, true)) {
        frame = exe_ctx.GetFramePtr();
        if (frame)
          frame_block = frame->GetFrameBlock();
      } else if (StateIsRunningState(state)) {
        return true; // Don't do any updating when we are running
      }
    }

    ValueObjectList local_values;
    if (frame_block) {
      // Only update the variables if they have changed
      if (m_frame_block != frame_block) {
        m_frame_block = frame_block;

        VariableList *locals = frame->GetVariableList(true);
        if (locals) {
          const DynamicValueType use_dynamic = eDynamicDontRunTarget;
          for (const VariableSP &local_sp : *locals) {
            ValueObjectSP value_sp =
                frame->GetValueObjectForFrameVariable(local_sp, use_dynamic);
            if (value_sp) {
              ValueObjectSP synthetic_value_sp = value_sp->GetSyntheticValue();
              if (synthetic_value_sp)
                local_values.Append(synthetic_value_sp);
              else
                local_values.Append(value_sp);
            }
          }
          // Update the values
          SetValues(local_values);
        }
      }
    } else {
      m_frame_block = nullptr;
      // Update the values with an empty list if there is no frame
      SetValues(local_values);
    }

    return ValueObjectListDelegate::WindowDelegateDraw(window, force);
  }

protected:
  Debugger &m_debugger;
  Block *m_frame_block;
};

class RegistersWindowDelegate : public ValueObjectListDelegate {
public:
  RegistersWindowDelegate(Debugger &debugger)
      : ValueObjectListDelegate(), m_debugger(debugger) {}

  ~RegistersWindowDelegate() override = default;

  const char *WindowDelegateGetHelpText() override {
    return "Register window keyboard shortcuts:";
  }

  bool WindowDelegateDraw(Window &window, bool force) override {
    ExecutionContext exe_ctx(
        m_debugger.GetCommandInterpreter().GetExecutionContext());
    StackFrame *frame = exe_ctx.GetFramePtr();

    ValueObjectList value_list;
    if (frame) {
      if (frame->GetStackID() != m_stack_id) {
        m_stack_id = frame->GetStackID();
        RegisterContextSP reg_ctx(frame->GetRegisterContext());
        if (reg_ctx) {
          const uint32_t num_sets = reg_ctx->GetRegisterSetCount();
          for (uint32_t set_idx = 0; set_idx < num_sets; ++set_idx) {
            value_list.Append(
                ValueObjectRegisterSet::Create(frame, reg_ctx, set_idx));
          }
        }
        SetValues(value_list);
      }
    } else {
      Process *process = exe_ctx.GetProcessPtr();
      if (process && process->IsAlive())
        return true; // Don't do any updating if we are running
      else {
        // Update the values with an empty list if there is no process or the
        // process isn't alive anymore
        SetValues(value_list);
      }
    }
    return ValueObjectListDelegate::WindowDelegateDraw(window, force);
  }

protected:
  Debugger &m_debugger;
  StackID m_stack_id;
};

static const char *CursesKeyToCString(int ch) {
  static char g_desc[32];
  if (ch >= KEY_F0 && ch < KEY_F0 + 64) {
    snprintf(g_desc, sizeof(g_desc), "F%u", ch - KEY_F0);
    return g_desc;
  }
  switch (ch) {
  case KEY_DOWN:
    return "down";
  case KEY_UP:
    return "up";
  case KEY_LEFT:
    return "left";
  case KEY_RIGHT:
    return "right";
  case KEY_HOME:
    return "home";
  case KEY_BACKSPACE:
    return "backspace";
  case KEY_DL:
    return "delete-line";
  case KEY_IL:
    return "insert-line";
  case KEY_DC:
    return "delete-char";
  case KEY_IC:
    return "insert-char";
  case KEY_CLEAR:
    return "clear";
  case KEY_EOS:
    return "clear-to-eos";
  case KEY_EOL:
    return "clear-to-eol";
  case KEY_SF:
    return "scroll-forward";
  case KEY_SR:
    return "scroll-backward";
  case KEY_NPAGE:
    return "page-down";
  case KEY_PPAGE:
    return "page-up";
  case KEY_STAB:
    return "set-tab";
  case KEY_CTAB:
    return "clear-tab";
  case KEY_CATAB:
    return "clear-all-tabs";
  case KEY_ENTER:
    return "enter";
  case KEY_PRINT:
    return "print";
  case KEY_LL:
    return "lower-left key";
  case KEY_A1:
    return "upper left of keypad";
  case KEY_A3:
    return "upper right of keypad";
  case KEY_B2:
    return "center of keypad";
  case KEY_C1:
    return "lower left of keypad";
  case KEY_C3:
    return "lower right of keypad";
  case KEY_BTAB:
    return "back-tab key";
  case KEY_BEG:
    return "begin key";
  case KEY_CANCEL:
    return "cancel key";
  case KEY_CLOSE:
    return "close key";
  case KEY_COMMAND:
    return "command key";
  case KEY_COPY:
    return "copy key";
  case KEY_CREATE:
    return "create key";
  case KEY_END:
    return "end key";
  case KEY_EXIT:
    return "exit key";
  case KEY_FIND:
    return "find key";
  case KEY_HELP:
    return "help key";
  case KEY_MARK:
    return "mark key";
  case KEY_MESSAGE:
    return "message key";
  case KEY_MOVE:
    return "move key";
  case KEY_NEXT:
    return "next key";
  case KEY_OPEN:
    return "open key";
  case KEY_OPTIONS:
    return "options key";
  case KEY_PREVIOUS:
    return "previous key";
  case KEY_REDO:
    return "redo key";
  case KEY_REFERENCE:
    return "reference key";
  case KEY_REFRESH:
    return "refresh key";
  case KEY_REPLACE:
    return "replace key";
  case KEY_RESTART:
    return "restart key";
  case KEY_RESUME:
    return "resume key";
  case KEY_SAVE:
    return "save key";
  case KEY_SBEG:
    return "shifted begin key";
  case KEY_SCANCEL:
    return "shifted cancel key";
  case KEY_SCOMMAND:
    return "shifted command key";
  case KEY_SCOPY:
    return "shifted copy key";
  case KEY_SCREATE:
    return "shifted create key";
  case KEY_SDC:
    return "shifted delete-character key";
  case KEY_SDL:
    return "shifted delete-line key";
  case KEY_SELECT:
    return "select key";
  case KEY_SEND:
    return "shifted end key";
  case KEY_SEOL:
    return "shifted clear-to-end-of-line key";
  case KEY_SEXIT:
    return "shifted exit key";
  case KEY_SFIND:
    return "shifted find key";
  case KEY_SHELP:
    return "shifted help key";
  case KEY_SHOME:
    return "shifted home key";
  case KEY_SIC:
    return "shifted insert-character key";
  case KEY_SLEFT:
    return "shifted left-arrow key";
  case KEY_SMESSAGE:
    return "shifted message key";
  case KEY_SMOVE:
    return "shifted move key";
  case KEY_SNEXT:
    return "shifted next key";
  case KEY_SOPTIONS:
    return "shifted options key";
  case KEY_SPREVIOUS:
    return "shifted previous key";
  case KEY_SPRINT:
    return "shifted print key";
  case KEY_SREDO:
    return "shifted redo key";
  case KEY_SREPLACE:
    return "shifted replace key";
  case KEY_SRIGHT:
    return "shifted right-arrow key";
  case KEY_SRSUME:
    return "shifted resume key";
  case KEY_SSAVE:
    return "shifted save key";
  case KEY_SSUSPEND:
    return "shifted suspend key";
  case KEY_SUNDO:
    return "shifted undo key";
  case KEY_SUSPEND:
    return "suspend key";
  case KEY_UNDO:
    return "undo key";
  case KEY_MOUSE:
    return "Mouse event has occurred";
  case KEY_RESIZE:
    return "Terminal resize event";
#ifdef KEY_EVENT
  case KEY_EVENT:
    return "We were interrupted by an event";
#endif
  case KEY_RETURN:
    return "return";
  case ' ':
    return "space";
  case '\t':
    return "tab";
  case KEY_ESCAPE:
    return "escape";
  default:
    if (llvm::isPrint(ch))
      snprintf(g_desc, sizeof(g_desc), "%c", ch);
    else
      snprintf(g_desc, sizeof(g_desc), "\\x%2.2x", ch);
    return g_desc;
  }
  return nullptr;
}

HelpDialogDelegate::HelpDialogDelegate(const char *text,
                                       KeyHelp *key_help_array)
    : m_text(), m_first_visible_line(0) {
  if (text && text[0]) {
    m_text.SplitIntoLines(text);
    m_text.AppendString("");
  }
  if (key_help_array) {
    for (KeyHelp *key = key_help_array; key->ch; ++key) {
      StreamString key_description;
      key_description.Printf("%10s - %s", CursesKeyToCString(key->ch),
                             key->description);
      m_text.AppendString(key_description.GetString());
    }
  }
}

HelpDialogDelegate::~HelpDialogDelegate() = default;

bool HelpDialogDelegate::WindowDelegateDraw(Window &window, bool force) {
  window.Erase();
  const int window_height = window.GetHeight();
  int x = 2;
  int y = 1;
  const int min_y = y;
  const int max_y = window_height - 1 - y;
  const size_t num_visible_lines = max_y - min_y + 1;
  const size_t num_lines = m_text.GetSize();
  const char *bottom_message;
  if (num_lines <= num_visible_lines)
    bottom_message = "Press any key to exit";
  else
    bottom_message = "Use arrows to scroll, any other key to exit";
  window.DrawTitleBox(window.GetName(), bottom_message);
  while (y <= max_y) {
    window.MoveCursor(x, y);
    window.PutCStringTruncated(
        1, m_text.GetStringAtIndex(m_first_visible_line + y - min_y));
    ++y;
  }
  return true;
}

HandleCharResult HelpDialogDelegate::WindowDelegateHandleChar(Window &window,
                                                              int key) {
  bool done = false;
  const size_t num_lines = m_text.GetSize();
  const size_t num_visible_lines = window.GetHeight() - 2;

  if (num_lines <= num_visible_lines) {
    done = true;
    // If we have all lines visible and don't need scrolling, then any key
    // press will cause us to exit
  } else {
    switch (key) {
    case KEY_UP:
      if (m_first_visible_line > 0)
        --m_first_visible_line;
      break;

    case KEY_DOWN:
      if (m_first_visible_line + num_visible_lines < num_lines)
        ++m_first_visible_line;
      break;

    case KEY_PPAGE:
    case ',':
      if (m_first_visible_line > 0) {
        if (static_cast<size_t>(m_first_visible_line) >= num_visible_lines)
          m_first_visible_line -= num_visible_lines;
        else
          m_first_visible_line = 0;
      }
      break;

    case KEY_NPAGE:
    case '.':
      if (m_first_visible_line + num_visible_lines < num_lines) {
        m_first_visible_line += num_visible_lines;
        if (static_cast<size_t>(m_first_visible_line) > num_lines)
          m_first_visible_line = num_lines - num_visible_lines;
      }
      break;

    default:
      done = true;
      break;
    }
  }
  if (done)
    window.GetParent()->RemoveSubWindow(&window);
  return eKeyHandled;
}

class ApplicationDelegate : public WindowDelegate, public MenuDelegate {
public:
  enum {
    eMenuID_LLDB = 1,
    eMenuID_LLDBAbout,
    eMenuID_LLDBExit,

    eMenuID_Target,
    eMenuID_TargetCreate,
    eMenuID_TargetDelete,

    eMenuID_Process,
    eMenuID_ProcessAttach,
    eMenuID_ProcessDetachResume,
    eMenuID_ProcessDetachSuspended,
    eMenuID_ProcessLaunch,
    eMenuID_ProcessContinue,
    eMenuID_ProcessHalt,
    eMenuID_ProcessKill,

    eMenuID_Thread,
    eMenuID_ThreadStepIn,
    eMenuID_ThreadStepOver,
    eMenuID_ThreadStepOut,

    eMenuID_View,
    eMenuID_ViewBacktrace,
    eMenuID_ViewRegisters,
    eMenuID_ViewSource,
    eMenuID_ViewVariables,

    eMenuID_Help,
    eMenuID_HelpGUIHelp
  };

  ApplicationDelegate(Application &app, Debugger &debugger)
      : WindowDelegate(), MenuDelegate(), m_app(app), m_debugger(debugger) {}

  ~ApplicationDelegate() override = default;

  bool WindowDelegateDraw(Window &window, bool force) override {
    return false; // Drawing not handled, let standard window drawing happen
  }

  HandleCharResult WindowDelegateHandleChar(Window &window, int key) override {
    switch (key) {
    case '\t':
      window.SelectNextWindowAsActive();
      return eKeyHandled;

    case KEY_BTAB:
      window.SelectPreviousWindowAsActive();
      return eKeyHandled;

    case 'h':
      window.CreateHelpSubwindow();
      return eKeyHandled;

    case KEY_ESCAPE:
      return eQuitApplication;

    default:
      break;
    }
    return eKeyNotHandled;
  }

  const char *WindowDelegateGetHelpText() override {
    return "Welcome to the LLDB curses GUI.\n\n"
           "Press the TAB key to change the selected view.\n"
           "Each view has its own keyboard shortcuts, press 'h' to open a "
           "dialog to display them.\n\n"
           "Common key bindings for all views:";
  }

  KeyHelp *WindowDelegateGetKeyHelp() override {
    static curses::KeyHelp g_source_view_key_help[] = {
        {'\t', "Select next view"},
        {KEY_BTAB, "Select previous view"},
        {'h', "Show help dialog with view specific key bindings"},
        {',', "Page up"},
        {'.', "Page down"},
        {KEY_UP, "Select previous"},
        {KEY_DOWN, "Select next"},
        {KEY_LEFT, "Unexpand or select parent"},
        {KEY_RIGHT, "Expand"},
        {KEY_PPAGE, "Page up"},
        {KEY_NPAGE, "Page down"},
        {'\0', nullptr}};
    return g_source_view_key_help;
  }

  MenuActionResult MenuDelegateAction(Menu &menu) override {
    switch (menu.GetIdentifier()) {
    case eMenuID_ThreadStepIn: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasThreadScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive() &&
            StateIsStoppedState(process->GetState(), true))
          exe_ctx.GetThreadRef().StepIn(true);
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_ThreadStepOut: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasThreadScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive() &&
            StateIsStoppedState(process->GetState(), true))
          exe_ctx.GetThreadRef().StepOut();
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_ThreadStepOver: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasThreadScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive() &&
            StateIsStoppedState(process->GetState(), true))
          exe_ctx.GetThreadRef().StepOver(true);
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_ProcessContinue: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasProcessScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive() &&
            StateIsStoppedState(process->GetState(), true))
          process->Resume();
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_ProcessKill: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasProcessScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive())
          process->Destroy(false);
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_ProcessHalt: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasProcessScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive())
          process->Halt();
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_ProcessDetachResume:
    case eMenuID_ProcessDetachSuspended: {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasProcessScope()) {
        Process *process = exe_ctx.GetProcessPtr();
        if (process && process->IsAlive())
          process->Detach(menu.GetIdentifier() ==
                          eMenuID_ProcessDetachSuspended);
      }
    }
      return MenuActionResult::Handled;

    case eMenuID_Process: {
      // Populate the menu with all of the threads if the process is stopped
      // when the Process menu gets selected and is about to display its
      // submenu.
      Menus &submenus = menu.GetSubmenus();
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      Process *process = exe_ctx.GetProcessPtr();
      if (process && process->IsAlive() &&
          StateIsStoppedState(process->GetState(), true)) {
        if (submenus.size() == 7)
          menu.AddSubmenu(MenuSP(new Menu(Menu::Type::Separator)));
        else if (submenus.size() > 8)
          submenus.erase(submenus.begin() + 8, submenus.end());

        ThreadList &threads = process->GetThreadList();
        std::lock_guard<std::recursive_mutex> guard(threads.GetMutex());
        size_t num_threads = threads.GetSize();
        for (size_t i = 0; i < num_threads; ++i) {
          ThreadSP thread_sp = threads.GetThreadAtIndex(i);
          char menu_char = '\0';
          if (i < 9)
            menu_char = '1' + i;
          StreamString thread_menu_title;
          thread_menu_title.Printf("Thread %u", thread_sp->GetIndexID());
          const char *thread_name = thread_sp->GetName();
          if (thread_name && thread_name[0])
            thread_menu_title.Printf(" %s", thread_name);
          else {
            const char *queue_name = thread_sp->GetQueueName();
            if (queue_name && queue_name[0])
              thread_menu_title.Printf(" %s", queue_name);
          }
          menu.AddSubmenu(
              MenuSP(new Menu(thread_menu_title.GetString().str().c_str(),
                              nullptr, menu_char, thread_sp->GetID())));
        }
      } else if (submenus.size() > 7) {
        // Remove the separator and any other thread submenu items that were
        // previously added
        submenus.erase(submenus.begin() + 7, submenus.end());
      }
      // Since we are adding and removing items we need to recalculate the name
      // lengths
      menu.RecalculateNameLengths();
    }
      return MenuActionResult::Handled;

    case eMenuID_ViewVariables: {
      WindowSP main_window_sp = m_app.GetMainWindow();
      WindowSP source_window_sp = main_window_sp->FindSubWindow("Source");
      WindowSP variables_window_sp = main_window_sp->FindSubWindow("Variables");
      WindowSP registers_window_sp = main_window_sp->FindSubWindow("Registers");
      const Rect source_bounds = source_window_sp->GetBounds();

      if (variables_window_sp) {
        const Rect variables_bounds = variables_window_sp->GetBounds();

        main_window_sp->RemoveSubWindow(variables_window_sp.get());

        if (registers_window_sp) {
          // We have a registers window, so give all the area back to the
          // registers window
          Rect registers_bounds = variables_bounds;
          registers_bounds.size.width = source_bounds.size.width;
          registers_window_sp->SetBounds(registers_bounds);
        } else {
          // We have no registers window showing so give the bottom area back
          // to the source view
          source_window_sp->Resize(source_bounds.size.width,
                                   source_bounds.size.height +
                                       variables_bounds.size.height);
        }
      } else {
        Rect new_variables_rect;
        if (registers_window_sp) {
          // We have a registers window so split the area of the registers
          // window into two columns where the left hand side will be the
          // variables and the right hand side will be the registers
          const Rect variables_bounds = registers_window_sp->GetBounds();
          Rect new_registers_rect;
          variables_bounds.VerticalSplitPercentage(0.50, new_variables_rect,
                                                   new_registers_rect);
          registers_window_sp->SetBounds(new_registers_rect);
        } else {
          // No registers window, grab the bottom part of the source window
          Rect new_source_rect;
          source_bounds.HorizontalSplitPercentage(0.70, new_source_rect,
                                                  new_variables_rect);
          source_window_sp->SetBounds(new_source_rect);
        }
        WindowSP new_window_sp = main_window_sp->CreateSubWindow(
            "Variables", new_variables_rect, false);
        new_window_sp->SetDelegate(
            WindowDelegateSP(new FrameVariablesWindowDelegate(m_debugger)));
      }
      touchwin(stdscr);
    }
      return MenuActionResult::Handled;

    case eMenuID_ViewRegisters: {
      WindowSP main_window_sp = m_app.GetMainWindow();
      WindowSP source_window_sp = main_window_sp->FindSubWindow("Source");
      WindowSP variables_window_sp = main_window_sp->FindSubWindow("Variables");
      WindowSP registers_window_sp = main_window_sp->FindSubWindow("Registers");
      const Rect source_bounds = source_window_sp->GetBounds();

      if (registers_window_sp) {
        if (variables_window_sp) {
          const Rect variables_bounds = variables_window_sp->GetBounds();

          // We have a variables window, so give all the area back to the
          // variables window
          variables_window_sp->Resize(variables_bounds.size.width +
                                          registers_window_sp->GetWidth(),
                                      variables_bounds.size.height);
        } else {
          // We have no variables window showing so give the bottom area back
          // to the source view
          source_window_sp->Resize(source_bounds.size.width,
                                   source_bounds.size.height +
                                       registers_window_sp->GetHeight());
        }
        main_window_sp->RemoveSubWindow(registers_window_sp.get());
      } else {
        Rect new_regs_rect;
        if (variables_window_sp) {
          // We have a variables window, split it into two columns where the
          // left hand side will be the variables and the right hand side will
          // be the registers
          const Rect variables_bounds = variables_window_sp->GetBounds();
          Rect new_vars_rect;
          variables_bounds.VerticalSplitPercentage(0.50, new_vars_rect,
                                                   new_regs_rect);
          variables_window_sp->SetBounds(new_vars_rect);
        } else {
          // No variables window, grab the bottom part of the source window
          Rect new_source_rect;
          source_bounds.HorizontalSplitPercentage(0.70, new_source_rect,
                                                  new_regs_rect);
          source_window_sp->SetBounds(new_source_rect);
        }
        WindowSP new_window_sp =
            main_window_sp->CreateSubWindow("Registers", new_regs_rect, false);
        new_window_sp->SetDelegate(
            WindowDelegateSP(new RegistersWindowDelegate(m_debugger)));
      }
      touchwin(stdscr);
    }
      return MenuActionResult::Handled;

    case eMenuID_HelpGUIHelp:
      m_app.GetMainWindow()->CreateHelpSubwindow();
      return MenuActionResult::Handled;

    default:
      break;
    }

    return MenuActionResult::NotHandled;
  }

protected:
  Application &m_app;
  Debugger &m_debugger;
};

class StatusBarWindowDelegate : public WindowDelegate {
public:
  StatusBarWindowDelegate(Debugger &debugger) : m_debugger(debugger) {
    FormatEntity::Parse("Thread: ${thread.id%tid}", m_format);
  }

  ~StatusBarWindowDelegate() override = default;

  bool WindowDelegateDraw(Window &window, bool force) override {
    ExecutionContext exe_ctx =
        m_debugger.GetCommandInterpreter().GetExecutionContext();
    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = exe_ctx.GetThreadPtr();
    StackFrame *frame = exe_ctx.GetFramePtr();
    window.Erase();
    window.SetBackground(BlackOnWhite);
    window.MoveCursor(0, 0);
    if (process) {
      const StateType state = process->GetState();
      window.Printf("Process: %5" PRIu64 " %10s", process->GetID(),
                    StateAsCString(state));

      if (StateIsStoppedState(state, true)) {
        StreamString strm;
        if (thread && FormatEntity::Format(m_format, strm, nullptr, &exe_ctx,
                                           nullptr, nullptr, false, false)) {
          window.MoveCursor(40, 0);
          window.PutCStringTruncated(1, strm.GetString().str().c_str());
        }

        window.MoveCursor(60, 0);
        if (frame)
          window.Printf("Frame: %3u  PC = 0x%16.16" PRIx64,
                        frame->GetFrameIndex(),
                        frame->GetFrameCodeAddress().GetOpcodeLoadAddress(
                            exe_ctx.GetTargetPtr()));
      } else if (state == eStateExited) {
        const char *exit_desc = process->GetExitDescription();
        const int exit_status = process->GetExitStatus();
        if (exit_desc && exit_desc[0])
          window.Printf(" with status = %i (%s)", exit_status, exit_desc);
        else
          window.Printf(" with status = %i", exit_status);
      }
    }
    return true;
  }

protected:
  Debugger &m_debugger;
  FormatEntity::Entry m_format;
};

class SourceFileWindowDelegate : public WindowDelegate {
public:
  SourceFileWindowDelegate(Debugger &debugger)
      : WindowDelegate(), m_debugger(debugger), m_sc(), m_file_sp(),
        m_disassembly_scope(nullptr), m_disassembly_sp(), m_disassembly_range(),
        m_title(), m_line_width(4), m_selected_line(0), m_pc_line(0),
        m_stop_id(0), m_frame_idx(UINT32_MAX), m_first_visible_line(0),
        m_min_x(0), m_min_y(0), m_max_x(0), m_max_y(0) {}

  ~SourceFileWindowDelegate() override = default;

  void Update(const SymbolContext &sc) { m_sc = sc; }

  uint32_t NumVisibleLines() const { return m_max_y - m_min_y; }

  const char *WindowDelegateGetHelpText() override {
    return "Source/Disassembly window keyboard shortcuts:";
  }

  KeyHelp *WindowDelegateGetKeyHelp() override {
    static curses::KeyHelp g_source_view_key_help[] = {
        {KEY_RETURN, "Run to selected line with one shot breakpoint"},
        {KEY_UP, "Select previous source line"},
        {KEY_DOWN, "Select next source line"},
        {KEY_PPAGE, "Page up"},
        {KEY_NPAGE, "Page down"},
        {'b', "Set breakpoint on selected source/disassembly line"},
        {'c', "Continue process"},
        {'D', "Detach with process suspended"},
        {'h', "Show help dialog"},
        {'n', "Step over (source line)"},
        {'N', "Step over (single instruction)"},
        {'f', "Step out (finish)"},
        {'s', "Step in (source line)"},
        {'S', "Step in (single instruction)"},
        {'u', "Frame up"},
        {'d', "Frame down"},
        {',', "Page up"},
        {'.', "Page down"},
        {'\0', nullptr}};
    return g_source_view_key_help;
  }

  bool WindowDelegateDraw(Window &window, bool force) override {
    ExecutionContext exe_ctx =
        m_debugger.GetCommandInterpreter().GetExecutionContext();
    Process *process = exe_ctx.GetProcessPtr();
    Thread *thread = nullptr;

    bool update_location = false;
    if (process) {
      StateType state = process->GetState();
      if (StateIsStoppedState(state, true)) {
        // We are stopped, so it is ok to
        update_location = true;
      }
    }

    m_min_x = 1;
    m_min_y = 2;
    m_max_x = window.GetMaxX() - 1;
    m_max_y = window.GetMaxY() - 1;

    const uint32_t num_visible_lines = NumVisibleLines();
    StackFrameSP frame_sp;
    bool set_selected_line_to_pc = false;

    if (update_location) {
      const bool process_alive = process ? process->IsAlive() : false;
      bool thread_changed = false;
      if (process_alive) {
        thread = exe_ctx.GetThreadPtr();
        if (thread) {
          frame_sp = thread->GetSelectedFrame();
          auto tid = thread->GetID();
          thread_changed = tid != m_tid;
          m_tid = tid;
        } else {
          if (m_tid != LLDB_INVALID_THREAD_ID) {
            thread_changed = true;
            m_tid = LLDB_INVALID_THREAD_ID;
          }
        }
      }
      const uint32_t stop_id = process ? process->GetStopID() : 0;
      const bool stop_id_changed = stop_id != m_stop_id;
      bool frame_changed = false;
      m_stop_id = stop_id;
      m_title.Clear();
      if (frame_sp) {
        m_sc = frame_sp->GetSymbolContext(eSymbolContextEverything);
        if (m_sc.module_sp) {
          m_title.Printf(
              "%s", m_sc.module_sp->GetFileSpec().GetFilename().GetCString());
          ConstString func_name = m_sc.GetFunctionName();
          if (func_name)
            m_title.Printf("`%s", func_name.GetCString());
        }
        const uint32_t frame_idx = frame_sp->GetFrameIndex();
        frame_changed = frame_idx != m_frame_idx;
        m_frame_idx = frame_idx;
      } else {
        m_sc.Clear(true);
        frame_changed = m_frame_idx != UINT32_MAX;
        m_frame_idx = UINT32_MAX;
      }

      const bool context_changed =
          thread_changed || frame_changed || stop_id_changed;

      if (process_alive) {
        if (m_sc.line_entry.IsValid()) {
          m_pc_line = m_sc.line_entry.line;
          if (m_pc_line != UINT32_MAX)
            --m_pc_line; // Convert to zero based line number...
          // Update the selected line if the stop ID changed...
          if (context_changed)
            m_selected_line = m_pc_line;

          if (m_file_sp && m_file_sp->GetFileSpec() == m_sc.line_entry.file) {
            // Same file, nothing to do, we should either have the lines or not
            // (source file missing)
            if (m_selected_line >= static_cast<size_t>(m_first_visible_line)) {
              if (m_selected_line >= m_first_visible_line + num_visible_lines)
                m_first_visible_line = m_selected_line - 10;
            } else {
              if (m_selected_line > 10)
                m_first_visible_line = m_selected_line - 10;
              else
                m_first_visible_line = 0;
            }
          } else {
            // File changed, set selected line to the line with the PC
            m_selected_line = m_pc_line;
            m_file_sp =
                m_debugger.GetSourceManager().GetFile(m_sc.line_entry.file);
            if (m_file_sp) {
              const size_t num_lines = m_file_sp->GetNumLines();
              m_line_width = 1;
              for (size_t n = num_lines; n >= 10; n = n / 10)
                ++m_line_width;

              if (num_lines < num_visible_lines ||
                  m_selected_line < num_visible_lines)
                m_first_visible_line = 0;
              else
                m_first_visible_line = m_selected_line - 10;
            }
          }
        } else {
          m_file_sp.reset();
        }

        if (!m_file_sp || m_file_sp->GetNumLines() == 0) {
          // Show disassembly
          bool prefer_file_cache = false;
          if (m_sc.function) {
            if (m_disassembly_scope != m_sc.function) {
              m_disassembly_scope = m_sc.function;
              m_disassembly_sp = m_sc.function->GetInstructions(
                  exe_ctx, nullptr, prefer_file_cache);
              if (m_disassembly_sp) {
                set_selected_line_to_pc = true;
                m_disassembly_range = m_sc.function->GetAddressRange();
              } else {
                m_disassembly_range.Clear();
              }
            } else {
              set_selected_line_to_pc = context_changed;
            }
          } else if (m_sc.symbol) {
            if (m_disassembly_scope != m_sc.symbol) {
              m_disassembly_scope = m_sc.symbol;
              m_disassembly_sp = m_sc.symbol->GetInstructions(
                  exe_ctx, nullptr, prefer_file_cache);
              if (m_disassembly_sp) {
                set_selected_line_to_pc = true;
                m_disassembly_range.GetBaseAddress() =
                    m_sc.symbol->GetAddress();
                m_disassembly_range.SetByteSize(m_sc.symbol->GetByteSize());
              } else {
                m_disassembly_range.Clear();
              }
            } else {
              set_selected_line_to_pc = context_changed;
            }
          }
        }
      } else {
        m_pc_line = UINT32_MAX;
      }
    }

    const int window_width = window.GetWidth();
    window.Erase();
    window.DrawTitleBox("Sources");
    if (!m_title.GetString().empty()) {
      window.AttributeOn(A_REVERSE);
      window.MoveCursor(1, 1);
      window.PutChar(' ');
      window.PutCStringTruncated(1, m_title.GetString().str().c_str());
      int x = window.GetCursorX();
      if (x < window_width - 1) {
        window.Printf("%*s", window_width - x - 1, "");
      }
      window.AttributeOff(A_REVERSE);
    }

    Target *target = exe_ctx.GetTargetPtr();
    const size_t num_source_lines = GetNumSourceLines();
    if (num_source_lines > 0) {
      // Display source
      BreakpointLines bp_lines;
      if (target) {
        BreakpointList &bp_list = target->GetBreakpointList();
        const size_t num_bps = bp_list.GetSize();
        for (size_t bp_idx = 0; bp_idx < num_bps; ++bp_idx) {
          BreakpointSP bp_sp = bp_list.GetBreakpointAtIndex(bp_idx);
          const size_t num_bps_locs = bp_sp->GetNumLocations();
          for (size_t bp_loc_idx = 0; bp_loc_idx < num_bps_locs; ++bp_loc_idx) {
            BreakpointLocationSP bp_loc_sp =
                bp_sp->GetLocationAtIndex(bp_loc_idx);
            LineEntry bp_loc_line_entry;
            if (bp_loc_sp->GetAddress().CalculateSymbolContextLineEntry(
                    bp_loc_line_entry)) {
              if (m_file_sp->GetFileSpec() == bp_loc_line_entry.file) {
                bp_lines.insert(bp_loc_line_entry.line);
              }
            }
          }
        }
      }

      const attr_t selected_highlight_attr = A_REVERSE;
      const attr_t pc_highlight_attr = COLOR_PAIR(BlackOnBlue);

      for (size_t i = 0; i < num_visible_lines; ++i) {
        const uint32_t curr_line = m_first_visible_line + i;
        if (curr_line < num_source_lines) {
          const int line_y = m_min_y + i;
          window.MoveCursor(1, line_y);
          const bool is_pc_line = curr_line == m_pc_line;
          const bool line_is_selected = m_selected_line == curr_line;
          // Highlight the line as the PC line first, then if the selected line
          // isn't the same as the PC line, highlight it differently
          attr_t highlight_attr = 0;
          attr_t bp_attr = 0;
          if (is_pc_line)
            highlight_attr = pc_highlight_attr;
          else if (line_is_selected)
            highlight_attr = selected_highlight_attr;

          if (bp_lines.find(curr_line + 1) != bp_lines.end())
            bp_attr = COLOR_PAIR(BlackOnWhite);

          if (bp_attr)
            window.AttributeOn(bp_attr);

          window.Printf(" %*u ", m_line_width, curr_line + 1);

          if (bp_attr)
            window.AttributeOff(bp_attr);

          window.PutChar(ACS_VLINE);
          // Mark the line with the PC with a diamond
          if (is_pc_line)
            window.PutChar(ACS_DIAMOND);
          else
            window.PutChar(' ');

          if (highlight_attr)
            window.AttributeOn(highlight_attr);

          StreamString lineStream;
          m_file_sp->DisplaySourceLines(curr_line + 1, {}, 0, 0, &lineStream);
          StringRef line = lineStream.GetString();
          if (line.endswith("\n"))
            line = line.drop_back();
          window.OutputColoredStringTruncated(1, line, is_pc_line);

          if (is_pc_line && frame_sp &&
              frame_sp->GetConcreteFrameIndex() == 0) {
            StopInfoSP stop_info_sp;
            if (thread)
              stop_info_sp = thread->GetStopInfo();
            if (stop_info_sp) {
              const char *stop_description = stop_info_sp->GetDescription();
              if (stop_description && stop_description[0]) {
                size_t stop_description_len = strlen(stop_description);
                int desc_x = window_width - stop_description_len - 16;
                if (desc_x - window.GetCursorX() > 0)
                  window.Printf("%*s", desc_x - window.GetCursorX(), "");
                window.MoveCursor(window_width - stop_description_len - 16,
                                  line_y);
                const attr_t stop_reason_attr = COLOR_PAIR(WhiteOnBlue);
                window.AttributeOn(stop_reason_attr);
                window.PrintfTruncated(1, " <<< Thread %u: %s ",
                                       thread->GetIndexID(), stop_description);
                window.AttributeOff(stop_reason_attr);
              }
            } else {
              window.Printf("%*s", window_width - window.GetCursorX() - 1, "");
            }
          }
          if (highlight_attr)
            window.AttributeOff(highlight_attr);
        } else {
          break;
        }
      }
    } else {
      size_t num_disassembly_lines = GetNumDisassemblyLines();
      if (num_disassembly_lines > 0) {
        // Display disassembly
        BreakpointAddrs bp_file_addrs;
        Target *target = exe_ctx.GetTargetPtr();
        if (target) {
          BreakpointList &bp_list = target->GetBreakpointList();
          const size_t num_bps = bp_list.GetSize();
          for (size_t bp_idx = 0; bp_idx < num_bps; ++bp_idx) {
            BreakpointSP bp_sp = bp_list.GetBreakpointAtIndex(bp_idx);
            const size_t num_bps_locs = bp_sp->GetNumLocations();
            for (size_t bp_loc_idx = 0; bp_loc_idx < num_bps_locs;
                 ++bp_loc_idx) {
              BreakpointLocationSP bp_loc_sp =
                  bp_sp->GetLocationAtIndex(bp_loc_idx);
              LineEntry bp_loc_line_entry;
              const lldb::addr_t file_addr =
                  bp_loc_sp->GetAddress().GetFileAddress();
              if (file_addr != LLDB_INVALID_ADDRESS) {
                if (m_disassembly_range.ContainsFileAddress(file_addr))
                  bp_file_addrs.insert(file_addr);
              }
            }
          }
        }

        const attr_t selected_highlight_attr = A_REVERSE;
        const attr_t pc_highlight_attr = COLOR_PAIR(WhiteOnBlue);

        StreamString strm;

        InstructionList &insts = m_disassembly_sp->GetInstructionList();
        Address pc_address;

        if (frame_sp)
          pc_address = frame_sp->GetFrameCodeAddress();
        const uint32_t pc_idx =
            pc_address.IsValid()
                ? insts.GetIndexOfInstructionAtAddress(pc_address)
                : UINT32_MAX;
        if (set_selected_line_to_pc) {
          m_selected_line = pc_idx;
        }

        const uint32_t non_visible_pc_offset = (num_visible_lines / 5);
        if (static_cast<size_t>(m_first_visible_line) >= num_disassembly_lines)
          m_first_visible_line = 0;

        if (pc_idx < num_disassembly_lines) {
          if (pc_idx < static_cast<uint32_t>(m_first_visible_line) ||
              pc_idx >= m_first_visible_line + num_visible_lines)
            m_first_visible_line = pc_idx - non_visible_pc_offset;
        }

        for (size_t i = 0; i < num_visible_lines; ++i) {
          const uint32_t inst_idx = m_first_visible_line + i;
          Instruction *inst = insts.GetInstructionAtIndex(inst_idx).get();
          if (!inst)
            break;

          const int line_y = m_min_y + i;
          window.MoveCursor(1, line_y);
          const bool is_pc_line = frame_sp && inst_idx == pc_idx;
          const bool line_is_selected = m_selected_line == inst_idx;
          // Highlight the line as the PC line first, then if the selected line
          // isn't the same as the PC line, highlight it differently
          attr_t highlight_attr = 0;
          attr_t bp_attr = 0;
          if (is_pc_line)
            highlight_attr = pc_highlight_attr;
          else if (line_is_selected)
            highlight_attr = selected_highlight_attr;

          if (bp_file_addrs.find(inst->GetAddress().GetFileAddress()) !=
              bp_file_addrs.end())
            bp_attr = COLOR_PAIR(BlackOnWhite);

          if (bp_attr)
            window.AttributeOn(bp_attr);

          window.Printf(" 0x%16.16llx ",
                        static_cast<unsigned long long>(
                            inst->GetAddress().GetLoadAddress(target)));

          if (bp_attr)
            window.AttributeOff(bp_attr);

          window.PutChar(ACS_VLINE);
          // Mark the line with the PC with a diamond
          if (is_pc_line)
            window.PutChar(ACS_DIAMOND);
          else
            window.PutChar(' ');

          if (highlight_attr)
            window.AttributeOn(highlight_attr);

          const char *mnemonic = inst->GetMnemonic(&exe_ctx);
          const char *operands = inst->GetOperands(&exe_ctx);
          const char *comment = inst->GetComment(&exe_ctx);

          if (mnemonic != nullptr && mnemonic[0] == '\0')
            mnemonic = nullptr;
          if (operands != nullptr && operands[0] == '\0')
            operands = nullptr;
          if (comment != nullptr && comment[0] == '\0')
            comment = nullptr;

          strm.Clear();

          if (mnemonic != nullptr && operands != nullptr && comment != nullptr)
            strm.Printf("%-8s %-25s ; %s", mnemonic, operands, comment);
          else if (mnemonic != nullptr && operands != nullptr)
            strm.Printf("%-8s %s", mnemonic, operands);
          else if (mnemonic != nullptr)
            strm.Printf("%s", mnemonic);

          int right_pad = 1;
          window.PutCStringTruncated(right_pad, strm.GetData());

          if (is_pc_line && frame_sp &&
              frame_sp->GetConcreteFrameIndex() == 0) {
            StopInfoSP stop_info_sp;
            if (thread)
              stop_info_sp = thread->GetStopInfo();
            if (stop_info_sp) {
              const char *stop_description = stop_info_sp->GetDescription();
              if (stop_description && stop_description[0]) {
                size_t stop_description_len = strlen(stop_description);
                int desc_x = window_width - stop_description_len - 16;
                if (desc_x - window.GetCursorX() > 0)
                  window.Printf("%*s", desc_x - window.GetCursorX(), "");
                window.MoveCursor(window_width - stop_description_len - 15,
                                  line_y);
                window.PrintfTruncated(1, "<<< Thread %u: %s ",
                                       thread->GetIndexID(), stop_description);
              }
            } else {
              window.Printf("%*s", window_width - window.GetCursorX() - 1, "");
            }
          }
          if (highlight_attr)
            window.AttributeOff(highlight_attr);
        }
      }
    }
    return true; // Drawing handled
  }

  size_t GetNumLines() {
    size_t num_lines = GetNumSourceLines();
    if (num_lines == 0)
      num_lines = GetNumDisassemblyLines();
    return num_lines;
  }

  size_t GetNumSourceLines() const {
    if (m_file_sp)
      return m_file_sp->GetNumLines();
    return 0;
  }

  size_t GetNumDisassemblyLines() const {
    if (m_disassembly_sp)
      return m_disassembly_sp->GetInstructionList().GetSize();
    return 0;
  }

  HandleCharResult WindowDelegateHandleChar(Window &window, int c) override {
    const uint32_t num_visible_lines = NumVisibleLines();
    const size_t num_lines = GetNumLines();

    switch (c) {
    case ',':
    case KEY_PPAGE:
      // Page up key
      if (static_cast<uint32_t>(m_first_visible_line) > num_visible_lines)
        m_first_visible_line -= num_visible_lines;
      else
        m_first_visible_line = 0;
      m_selected_line = m_first_visible_line;
      return eKeyHandled;

    case '.':
    case KEY_NPAGE:
      // Page down key
      {
        if (m_first_visible_line + num_visible_lines < num_lines)
          m_first_visible_line += num_visible_lines;
        else if (num_lines < num_visible_lines)
          m_first_visible_line = 0;
        else
          m_first_visible_line = num_lines - num_visible_lines;
        m_selected_line = m_first_visible_line;
      }
      return eKeyHandled;

    case KEY_UP:
      if (m_selected_line > 0) {
        m_selected_line--;
        if (static_cast<size_t>(m_first_visible_line) > m_selected_line)
          m_first_visible_line = m_selected_line;
      }
      return eKeyHandled;

    case KEY_DOWN:
      if (m_selected_line + 1 < num_lines) {
        m_selected_line++;
        if (m_first_visible_line + num_visible_lines < m_selected_line)
          m_first_visible_line++;
      }
      return eKeyHandled;

    case '\r':
    case '\n':
    case KEY_ENTER:
      // Set a breakpoint and run to the line using a one shot breakpoint
      if (GetNumSourceLines() > 0) {
        ExecutionContext exe_ctx =
            m_debugger.GetCommandInterpreter().GetExecutionContext();
        if (exe_ctx.HasProcessScope() && exe_ctx.GetProcessRef().IsAlive()) {
          BreakpointSP bp_sp = exe_ctx.GetTargetRef().CreateBreakpoint(
              nullptr, // Don't limit the breakpoint to certain modules
              m_file_sp->GetFileSpec(), // Source file
              m_selected_line +
                  1, // Source line number (m_selected_line is zero based)
              0,     // Unspecified column.
              0,     // No offset
              eLazyBoolCalculate,  // Check inlines using global setting
              eLazyBoolCalculate,  // Skip prologue using global setting,
              false,               // internal
              false,               // request_hardware
              eLazyBoolCalculate); // move_to_nearest_code
          // Make breakpoint one shot
          bp_sp->GetOptions()->SetOneShot(true);
          exe_ctx.GetProcessRef().Resume();
        }
      } else if (m_selected_line < GetNumDisassemblyLines()) {
        const Instruction *inst = m_disassembly_sp->GetInstructionList()
                                      .GetInstructionAtIndex(m_selected_line)
                                      .get();
        ExecutionContext exe_ctx =
            m_debugger.GetCommandInterpreter().GetExecutionContext();
        if (exe_ctx.HasTargetScope()) {
          Address addr = inst->GetAddress();
          BreakpointSP bp_sp = exe_ctx.GetTargetRef().CreateBreakpoint(
              addr,   // lldb_private::Address
              false,  // internal
              false); // request_hardware
          // Make breakpoint one shot
          bp_sp->GetOptions()->SetOneShot(true);
          exe_ctx.GetProcessRef().Resume();
        }
      }
      return eKeyHandled;

    case 'b': // 'b' == toggle breakpoint on currently selected line
      ToggleBreakpointOnSelectedLine();
      return eKeyHandled;

    case 'D': // 'D' == detach and keep stopped
    {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasProcessScope())
        exe_ctx.GetProcessRef().Detach(true);
    }
      return eKeyHandled;

    case 'c':
      // 'c' == continue
      {
        ExecutionContext exe_ctx =
            m_debugger.GetCommandInterpreter().GetExecutionContext();
        if (exe_ctx.HasProcessScope())
          exe_ctx.GetProcessRef().Resume();
      }
      return eKeyHandled;

    case 'f':
      // 'f' == step out (finish)
      {
        ExecutionContext exe_ctx =
            m_debugger.GetCommandInterpreter().GetExecutionContext();
        if (exe_ctx.HasThreadScope() &&
            StateIsStoppedState(exe_ctx.GetProcessRef().GetState(), true)) {
          exe_ctx.GetThreadRef().StepOut();
        }
      }
      return eKeyHandled;

    case 'n': // 'n' == step over
    case 'N': // 'N' == step over instruction
    {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasThreadScope() &&
          StateIsStoppedState(exe_ctx.GetProcessRef().GetState(), true)) {
        bool source_step = (c == 'n');
        exe_ctx.GetThreadRef().StepOver(source_step);
      }
    }
      return eKeyHandled;

    case 's': // 's' == step into
    case 'S': // 'S' == step into instruction
    {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasThreadScope() &&
          StateIsStoppedState(exe_ctx.GetProcessRef().GetState(), true)) {
        bool source_step = (c == 's');
        exe_ctx.GetThreadRef().StepIn(source_step);
      }
    }
      return eKeyHandled;

    case 'u': // 'u' == frame up
    case 'd': // 'd' == frame down
    {
      ExecutionContext exe_ctx =
          m_debugger.GetCommandInterpreter().GetExecutionContext();
      if (exe_ctx.HasThreadScope()) {
        Thread *thread = exe_ctx.GetThreadPtr();
        uint32_t frame_idx = thread->GetSelectedFrameIndex();
        if (frame_idx == UINT32_MAX)
          frame_idx = 0;
        if (c == 'u' && frame_idx + 1 < thread->GetStackFrameCount())
          ++frame_idx;
        else if (c == 'd' && frame_idx > 0)
          --frame_idx;
        if (thread->SetSelectedFrameByIndex(frame_idx, true))
          exe_ctx.SetFrameSP(thread->GetSelectedFrame());
      }
    }
      return eKeyHandled;

    case 'h':
      window.CreateHelpSubwindow();
      return eKeyHandled;

    default:
      break;
    }
    return eKeyNotHandled;
  }

  void ToggleBreakpointOnSelectedLine() {
    ExecutionContext exe_ctx =
        m_debugger.GetCommandInterpreter().GetExecutionContext();
    if (!exe_ctx.HasTargetScope())
      return;
    if (GetNumSourceLines() > 0) {
      // Source file breakpoint.
      BreakpointList &bp_list = exe_ctx.GetTargetRef().GetBreakpointList();
      const size_t num_bps = bp_list.GetSize();
      for (size_t bp_idx = 0; bp_idx < num_bps; ++bp_idx) {
        BreakpointSP bp_sp = bp_list.GetBreakpointAtIndex(bp_idx);
        const size_t num_bps_locs = bp_sp->GetNumLocations();
        for (size_t bp_loc_idx = 0; bp_loc_idx < num_bps_locs; ++bp_loc_idx) {
          BreakpointLocationSP bp_loc_sp =
              bp_sp->GetLocationAtIndex(bp_loc_idx);
          LineEntry bp_loc_line_entry;
          if (bp_loc_sp->GetAddress().CalculateSymbolContextLineEntry(
                  bp_loc_line_entry)) {
            if (m_file_sp->GetFileSpec() == bp_loc_line_entry.file &&
                m_selected_line + 1 == bp_loc_line_entry.line) {
              bool removed =
                  exe_ctx.GetTargetRef().RemoveBreakpointByID(bp_sp->GetID());
              assert(removed);
              UNUSED_IF_ASSERT_DISABLED(removed);
              return; // Existing breakpoint removed.
            }
          }
        }
      }
      // No breakpoint found on the location, add it.
      BreakpointSP bp_sp = exe_ctx.GetTargetRef().CreateBreakpoint(
          nullptr, // Don't limit the breakpoint to certain modules
          m_file_sp->GetFileSpec(), // Source file
          m_selected_line +
              1, // Source line number (m_selected_line is zero based)
          0,     // No column specified.
          0,     // No offset
          eLazyBoolCalculate,  // Check inlines using global setting
          eLazyBoolCalculate,  // Skip prologue using global setting,
          false,               // internal
          false,               // request_hardware
          eLazyBoolCalculate); // move_to_nearest_code
    } else {
      // Disassembly breakpoint.
      assert(GetNumDisassemblyLines() > 0);
      assert(m_selected_line < GetNumDisassemblyLines());
      const Instruction *inst = m_disassembly_sp->GetInstructionList()
                                    .GetInstructionAtIndex(m_selected_line)
                                    .get();
      Address addr = inst->GetAddress();
      // Try to find it.
      BreakpointList &bp_list = exe_ctx.GetTargetRef().GetBreakpointList();
      const size_t num_bps = bp_list.GetSize();
      for (size_t bp_idx = 0; bp_idx < num_bps; ++bp_idx) {
        BreakpointSP bp_sp = bp_list.GetBreakpointAtIndex(bp_idx);
        const size_t num_bps_locs = bp_sp->GetNumLocations();
        for (size_t bp_loc_idx = 0; bp_loc_idx < num_bps_locs; ++bp_loc_idx) {
          BreakpointLocationSP bp_loc_sp =
              bp_sp->GetLocationAtIndex(bp_loc_idx);
          LineEntry bp_loc_line_entry;
          const lldb::addr_t file_addr =
              bp_loc_sp->GetAddress().GetFileAddress();
          if (file_addr == addr.GetFileAddress()) {
            bool removed =
                exe_ctx.GetTargetRef().RemoveBreakpointByID(bp_sp->GetID());
            assert(removed);
            UNUSED_IF_ASSERT_DISABLED(removed);
            return; // Existing breakpoint removed.
          }
        }
      }
      // No breakpoint found on the address, add it.
      BreakpointSP bp_sp =
          exe_ctx.GetTargetRef().CreateBreakpoint(addr, // lldb_private::Address
                                                  false,  // internal
                                                  false); // request_hardware
    }
  }

protected:
  typedef std::set<uint32_t> BreakpointLines;
  typedef std::set<lldb::addr_t> BreakpointAddrs;

  Debugger &m_debugger;
  SymbolContext m_sc;
  SourceManager::FileSP m_file_sp;
  SymbolContextScope *m_disassembly_scope;
  lldb::DisassemblerSP m_disassembly_sp;
  AddressRange m_disassembly_range;
  StreamString m_title;
  lldb::user_id_t m_tid;
  int m_line_width;
  uint32_t m_selected_line; // The selected line
  uint32_t m_pc_line;       // The line with the PC
  uint32_t m_stop_id;
  uint32_t m_frame_idx;
  int m_first_visible_line;
  int m_min_x;
  int m_min_y;
  int m_max_x;
  int m_max_y;
};

DisplayOptions ValueObjectListDelegate::g_options = {true};

IOHandlerCursesGUI::IOHandlerCursesGUI(Debugger &debugger)
    : IOHandler(debugger, IOHandler::Type::Curses) {}

void IOHandlerCursesGUI::Activate() {
  IOHandler::Activate();
  if (!m_app_ap) {
    m_app_ap = std::make_unique<Application>(GetInputFILE(), GetOutputFILE());

    // This is both a window and a menu delegate
    std::shared_ptr<ApplicationDelegate> app_delegate_sp(
        new ApplicationDelegate(*m_app_ap, m_debugger));

    MenuDelegateSP app_menu_delegate_sp =
        std::static_pointer_cast<MenuDelegate>(app_delegate_sp);
    MenuSP lldb_menu_sp(
        new Menu("LLDB", "F1", KEY_F(1), ApplicationDelegate::eMenuID_LLDB));
    MenuSP exit_menuitem_sp(
        new Menu("Exit", nullptr, 'x', ApplicationDelegate::eMenuID_LLDBExit));
    exit_menuitem_sp->SetCannedResult(MenuActionResult::Quit);
    lldb_menu_sp->AddSubmenu(MenuSP(new Menu(
        "About LLDB", nullptr, 'a', ApplicationDelegate::eMenuID_LLDBAbout)));
    lldb_menu_sp->AddSubmenu(MenuSP(new Menu(Menu::Type::Separator)));
    lldb_menu_sp->AddSubmenu(exit_menuitem_sp);

    MenuSP target_menu_sp(new Menu("Target", "F2", KEY_F(2),
                                   ApplicationDelegate::eMenuID_Target));
    target_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Create", nullptr, 'c', ApplicationDelegate::eMenuID_TargetCreate)));
    target_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Delete", nullptr, 'd', ApplicationDelegate::eMenuID_TargetDelete)));

    MenuSP process_menu_sp(new Menu("Process", "F3", KEY_F(3),
                                    ApplicationDelegate::eMenuID_Process));
    process_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Attach", nullptr, 'a', ApplicationDelegate::eMenuID_ProcessAttach)));
    process_menu_sp->AddSubmenu(
        MenuSP(new Menu("Detach and resume", nullptr, 'd',
                        ApplicationDelegate::eMenuID_ProcessDetachResume)));
    process_menu_sp->AddSubmenu(
        MenuSP(new Menu("Detach suspended", nullptr, 's',
                        ApplicationDelegate::eMenuID_ProcessDetachSuspended)));
    process_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Launch", nullptr, 'l', ApplicationDelegate::eMenuID_ProcessLaunch)));
    process_menu_sp->AddSubmenu(MenuSP(new Menu(Menu::Type::Separator)));
    process_menu_sp->AddSubmenu(
        MenuSP(new Menu("Continue", nullptr, 'c',
                        ApplicationDelegate::eMenuID_ProcessContinue)));
    process_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Halt", nullptr, 'h', ApplicationDelegate::eMenuID_ProcessHalt)));
    process_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Kill", nullptr, 'k', ApplicationDelegate::eMenuID_ProcessKill)));

    MenuSP thread_menu_sp(new Menu("Thread", "F4", KEY_F(4),
                                   ApplicationDelegate::eMenuID_Thread));
    thread_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Step In", nullptr, 'i', ApplicationDelegate::eMenuID_ThreadStepIn)));
    thread_menu_sp->AddSubmenu(
        MenuSP(new Menu("Step Over", nullptr, 'v',
                        ApplicationDelegate::eMenuID_ThreadStepOver)));
    thread_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Step Out", nullptr, 'o', ApplicationDelegate::eMenuID_ThreadStepOut)));

    MenuSP view_menu_sp(
        new Menu("View", "F5", KEY_F(5), ApplicationDelegate::eMenuID_View));
    view_menu_sp->AddSubmenu(
        MenuSP(new Menu("Backtrace", nullptr, 'b',
                        ApplicationDelegate::eMenuID_ViewBacktrace)));
    view_menu_sp->AddSubmenu(
        MenuSP(new Menu("Registers", nullptr, 'r',
                        ApplicationDelegate::eMenuID_ViewRegisters)));
    view_menu_sp->AddSubmenu(MenuSP(new Menu(
        "Source", nullptr, 's', ApplicationDelegate::eMenuID_ViewSource)));
    view_menu_sp->AddSubmenu(
        MenuSP(new Menu("Variables", nullptr, 'v',
                        ApplicationDelegate::eMenuID_ViewVariables)));

    MenuSP help_menu_sp(
        new Menu("Help", "F6", KEY_F(6), ApplicationDelegate::eMenuID_Help));
    help_menu_sp->AddSubmenu(MenuSP(new Menu(
        "GUI Help", nullptr, 'g', ApplicationDelegate::eMenuID_HelpGUIHelp)));

    m_app_ap->Initialize();
    WindowSP &main_window_sp = m_app_ap->GetMainWindow();

    MenuSP menubar_sp(new Menu(Menu::Type::Bar));
    menubar_sp->AddSubmenu(lldb_menu_sp);
    menubar_sp->AddSubmenu(target_menu_sp);
    menubar_sp->AddSubmenu(process_menu_sp);
    menubar_sp->AddSubmenu(thread_menu_sp);
    menubar_sp->AddSubmenu(view_menu_sp);
    menubar_sp->AddSubmenu(help_menu_sp);
    menubar_sp->SetDelegate(app_menu_delegate_sp);

    Rect content_bounds = main_window_sp->GetFrame();
    Rect menubar_bounds = content_bounds.MakeMenuBar();
    Rect status_bounds = content_bounds.MakeStatusBar();
    Rect source_bounds;
    Rect variables_bounds;
    Rect threads_bounds;
    Rect source_variables_bounds;
    content_bounds.VerticalSplitPercentage(0.80, source_variables_bounds,
                                           threads_bounds);
    source_variables_bounds.HorizontalSplitPercentage(0.70, source_bounds,
                                                      variables_bounds);

    WindowSP menubar_window_sp =
        main_window_sp->CreateSubWindow("Menubar", menubar_bounds, false);
    // Let the menubar get keys if the active window doesn't handle the keys
    // that are typed so it can respond to menubar key presses.
    menubar_window_sp->SetCanBeActive(
        false); // Don't let the menubar become the active window
    menubar_window_sp->SetDelegate(menubar_sp);

    WindowSP source_window_sp(
        main_window_sp->CreateSubWindow("Source", source_bounds, true));
    WindowSP variables_window_sp(
        main_window_sp->CreateSubWindow("Variables", variables_bounds, false));
    WindowSP threads_window_sp(
        main_window_sp->CreateSubWindow("Threads", threads_bounds, false));
    WindowSP status_window_sp(
        main_window_sp->CreateSubWindow("Status", status_bounds, false));
    status_window_sp->SetCanBeActive(
        false); // Don't let the status bar become the active window
    main_window_sp->SetDelegate(
        std::static_pointer_cast<WindowDelegate>(app_delegate_sp));
    source_window_sp->SetDelegate(
        WindowDelegateSP(new SourceFileWindowDelegate(m_debugger)));
    variables_window_sp->SetDelegate(
        WindowDelegateSP(new FrameVariablesWindowDelegate(m_debugger)));
    TreeDelegateSP thread_delegate_sp(new ThreadsTreeDelegate(m_debugger));
    threads_window_sp->SetDelegate(WindowDelegateSP(
        new TreeWindowDelegate(m_debugger, thread_delegate_sp)));
    status_window_sp->SetDelegate(
        WindowDelegateSP(new StatusBarWindowDelegate(m_debugger)));

    // Show the main help window once the first time the curses GUI is launched
    static bool g_showed_help = false;
    if (!g_showed_help) {
      g_showed_help = true;
      main_window_sp->CreateHelpSubwindow();
    }

    // All colors with black background.
    init_pair(1, COLOR_BLACK, COLOR_BLACK);
    init_pair(2, COLOR_RED, COLOR_BLACK);
    init_pair(3, COLOR_GREEN, COLOR_BLACK);
    init_pair(4, COLOR_YELLOW, COLOR_BLACK);
    init_pair(5, COLOR_BLUE, COLOR_BLACK);
    init_pair(6, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(7, COLOR_CYAN, COLOR_BLACK);
    init_pair(8, COLOR_WHITE, COLOR_BLACK);
    // All colors with blue background.
    init_pair(9, COLOR_BLACK, COLOR_BLUE);
    init_pair(10, COLOR_RED, COLOR_BLUE);
    init_pair(11, COLOR_GREEN, COLOR_BLUE);
    init_pair(12, COLOR_YELLOW, COLOR_BLUE);
    init_pair(13, COLOR_BLUE, COLOR_BLUE);
    init_pair(14, COLOR_MAGENTA, COLOR_BLUE);
    init_pair(15, COLOR_CYAN, COLOR_BLUE);
    init_pair(16, COLOR_WHITE, COLOR_BLUE);
    // These must match the order in the color indexes enum.
    init_pair(17, COLOR_BLACK, COLOR_WHITE);
    init_pair(18, COLOR_MAGENTA, COLOR_WHITE);
    static_assert(LastColorPairIndex == 18, "Color indexes do not match.");
  }
}

void IOHandlerCursesGUI::Deactivate() { m_app_ap->Terminate(); }

void IOHandlerCursesGUI::Run() {
  m_app_ap->Run(m_debugger);
  SetIsDone(true);
}

IOHandlerCursesGUI::~IOHandlerCursesGUI() = default;

void IOHandlerCursesGUI::Cancel() {}

bool IOHandlerCursesGUI::Interrupt() { return false; }

void IOHandlerCursesGUI::GotEOF() {}

void IOHandlerCursesGUI::TerminalSizeChanged() {
  m_app_ap->TerminalSizeChanged();
}

#endif // LLDB_ENABLE_CURSES
