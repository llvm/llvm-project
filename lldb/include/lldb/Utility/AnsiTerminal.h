//===---------------------AnsiTerminal.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_ANSITERMINAL_H
#define LLDB_UTILITY_ANSITERMINAL_H

#define ANSI_FG_COLOR_BLACK 30
#define ANSI_FG_COLOR_RED 31
#define ANSI_FG_COLOR_GREEN 32
#define ANSI_FG_COLOR_YELLOW 33
#define ANSI_FG_COLOR_BLUE 34
#define ANSI_FG_COLOR_PURPLE 35
#define ANSI_FG_COLOR_CYAN 36
#define ANSI_FG_COLOR_WHITE 37

#define ANSI_FG_COLOR_BRIGHT_BLACK 90
#define ANSI_FG_COLOR_BRIGHT_RED 91
#define ANSI_FG_COLOR_BRIGHT_GREEN 92
#define ANSI_FG_COLOR_BRIGHT_YELLOW 93
#define ANSI_FG_COLOR_BRIGHT_BLUE 94
#define ANSI_FG_COLOR_BRIGHT_PURPLE 95
#define ANSI_FG_COLOR_BRIGHT_CYAN 96
#define ANSI_FG_COLOR_BRIGHT_WHITE 97

#define ANSI_BG_COLOR_BLACK 40
#define ANSI_BG_COLOR_RED 41
#define ANSI_BG_COLOR_GREEN 42
#define ANSI_BG_COLOR_YELLOW 43
#define ANSI_BG_COLOR_BLUE 44
#define ANSI_BG_COLOR_PURPLE 45
#define ANSI_BG_COLOR_CYAN 46
#define ANSI_BG_COLOR_WHITE 47

#define ANSI_BG_COLOR_BRIGHT_BLACK 100
#define ANSI_BG_COLOR_BRIGHT_RED 101
#define ANSI_BG_COLOR_BRIGHT_GREEN 102
#define ANSI_BG_COLOR_BRIGHT_YELLOW 103
#define ANSI_BG_COLOR_BRIGHT_BLUE 104
#define ANSI_BG_COLOR_BRIGHT_PURPLE 105
#define ANSI_BG_COLOR_BRIGHT_CYAN 106
#define ANSI_BG_COLOR_BRIGHT_WHITE 107

#define ANSI_SPECIAL_FRAMED 51
#define ANSI_SPECIAL_ENCIRCLED 52

#define ANSI_CTRL_NORMAL 0
#define ANSI_CTRL_BOLD 1
#define ANSI_CTRL_FAINT 2
#define ANSI_CTRL_ITALIC 3
#define ANSI_CTRL_UNDERLINE 4
#define ANSI_CTRL_SLOW_BLINK 5
#define ANSI_CTRL_FAST_BLINK 6
#define ANSI_CTRL_IMAGE_NEGATIVE 7
#define ANSI_CTRL_CONCEAL 8
#define ANSI_CTRL_CROSSED_OUT 9

#define ANSI_ESC_START "\033["
#define ANSI_ESC_END "m"

#define ANSI_STR(s) #s
#define ANSI_DEF_STR(s) ANSI_STR(s)

#define ANSI_ESCAPE1(s) ANSI_ESC_START ANSI_DEF_STR(s) ANSI_ESC_END

#define ANSI_1_CTRL(ctrl1) "\033["##ctrl1 ANSI_ESC_END
#define ANSI_2_CTRL(ctrl1, ctrl2) "\033["##ctrl1 ";"##ctrl2 ANSI_ESC_END

#define ANSI_ESC_START_LEN 2

// Cursor Position, set cursor to position [l, c] (default = [1, 1]).
#define ANSI_CSI_CUP(...) ANSI_ESC_START #__VA_ARGS__ "H"
// Reset cursor to position.
#define ANSI_CSI_RESET_CURSOR ANSI_CSI_CUP()
// Erase In Display.
#define ANSI_CSI_ED(opt) ANSI_ESC_START #opt "J"
// Erase complete viewport.
#define ANSI_CSI_ERASE_VIEWPORT ANSI_CSI_ED(2)
// Erase scrollback.
#define ANSI_CSI_ERASE_SCROLLBACK ANSI_CSI_ED(3)

// OSC (Operating System Commands)
// https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
#define OSC_ESCAPE_START "\033"
#define OSC_ESCAPE_END "\x07"

// https://conemu.github.io/en/AnsiEscapeCodes.html#ConEmu_specific_OSC
#define OSC_PROGRESS_REMOVE OSC_ESCAPE_START "]9;4;0;0" OSC_ESCAPE_END
#define OSC_PROGRESS_SHOW OSC_ESCAPE_START "]9;4;1;%u" OSC_ESCAPE_END
#define OSC_PROGRESS_ERROR OSC_ESCAPE_START "]9;4;2;%u" OSC_ESCAPE_END
#define OSC_PROGRESS_INDETERMINATE OSC_ESCAPE_START "]9;4;3;%u" OSC_ESCAPE_END

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Locale.h"

#include "lldb/Utility/Stream.h"

#include <string>

namespace lldb_private {

namespace ansi {

inline std::string FormatAnsiTerminalCodes(llvm::StringRef format,
                                           bool do_color = true) {
  // Convert "${ansi.XXX}" tokens to ansi values or clear them if do_color is
  // false.
  // clang-format off
  static const struct {
    const char *name;
    const char *value;
  } g_color_tokens[] = {
#define _TO_STR2(_val) #_val
#define _TO_STR(_val) _TO_STR2(_val)
      {"fg.black}",         ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BLACK) ANSI_ESC_END},
      {"fg.red}",           ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_RED) ANSI_ESC_END},
      {"fg.green}",         ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_GREEN) ANSI_ESC_END},
      {"fg.yellow}",        ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_YELLOW) ANSI_ESC_END},
      {"fg.blue}",          ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BLUE) ANSI_ESC_END},
      {"fg.purple}",        ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_PURPLE) ANSI_ESC_END},
      {"fg.cyan}",          ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_CYAN) ANSI_ESC_END},
      {"fg.white}",         ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_WHITE) ANSI_ESC_END},
      {"fg.bright.black}",  ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_BLACK) ANSI_ESC_END},
      {"fg.bright.red}",    ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_RED) ANSI_ESC_END},
      {"fg.bright.green}",  ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_GREEN) ANSI_ESC_END},
      {"fg.bright.yellow}", ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_YELLOW) ANSI_ESC_END},
      {"fg.bright.blue}",   ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_BLUE) ANSI_ESC_END},
      {"fg.bright.purple}", ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_PURPLE) ANSI_ESC_END},
      {"fg.bright.cyan}",   ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_CYAN) ANSI_ESC_END},
      {"fg.bright.white}",  ANSI_ESC_START _TO_STR(ANSI_FG_COLOR_BRIGHT_WHITE) ANSI_ESC_END},
      {"bg.black}",         ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BLACK) ANSI_ESC_END},
      {"bg.red}",           ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_RED) ANSI_ESC_END},
      {"bg.green}",         ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_GREEN) ANSI_ESC_END},
      {"bg.yellow}",        ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_YELLOW) ANSI_ESC_END},
      {"bg.blue}",          ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BLUE) ANSI_ESC_END},
      {"bg.purple}",        ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_PURPLE) ANSI_ESC_END},
      {"bg.cyan}",          ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_CYAN) ANSI_ESC_END},
      {"bg.white}",         ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_WHITE) ANSI_ESC_END},
      {"bg.bright.black}",  ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_BLACK) ANSI_ESC_END},
      {"bg.bright.red}",    ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_RED) ANSI_ESC_END},
      {"bg.bright.green}",  ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_GREEN) ANSI_ESC_END},
      {"bg.bright.yellow}", ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_YELLOW) ANSI_ESC_END},
      {"bg.bright.blue}",   ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_BLUE) ANSI_ESC_END},
      {"bg.bright.purple}", ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_PURPLE) ANSI_ESC_END},
      {"bg.bright.cyan}",   ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_CYAN) ANSI_ESC_END},
      {"bg.bright.white}",  ANSI_ESC_START _TO_STR(ANSI_BG_COLOR_BRIGHT_WHITE) ANSI_ESC_END},
      {"normal}",           ANSI_ESC_START _TO_STR(ANSI_CTRL_NORMAL) ANSI_ESC_END},
      {"bold}",             ANSI_ESC_START _TO_STR(ANSI_CTRL_BOLD) ANSI_ESC_END},
      {"faint}",            ANSI_ESC_START _TO_STR(ANSI_CTRL_FAINT) ANSI_ESC_END},
      {"italic}",           ANSI_ESC_START _TO_STR(ANSI_CTRL_ITALIC) ANSI_ESC_END},
      {"underline}",        ANSI_ESC_START _TO_STR(ANSI_CTRL_UNDERLINE) ANSI_ESC_END},
      {"slow-blink}",       ANSI_ESC_START _TO_STR(ANSI_CTRL_SLOW_BLINK) ANSI_ESC_END},
      {"fast-blink}",       ANSI_ESC_START _TO_STR(ANSI_CTRL_FAST_BLINK) ANSI_ESC_END},
      {"negative}",         ANSI_ESC_START _TO_STR(ANSI_CTRL_IMAGE_NEGATIVE) ANSI_ESC_END},
      {"conceal}",          ANSI_ESC_START _TO_STR(ANSI_CTRL_CONCEAL) ANSI_ESC_END},
      {"crossed-out}",      ANSI_ESC_START _TO_STR(ANSI_CTRL_CROSSED_OUT) ANSI_ESC_END},
#undef _TO_STR
#undef _TO_STR2
  };
  // clang-format on
  auto codes = llvm::ArrayRef(g_color_tokens);

  static const char tok_hdr[] = "${ansi.";

  std::string fmt;
  while (!format.empty()) {
    llvm::StringRef left, right;
    std::tie(left, right) = format.split(tok_hdr);

    fmt += left;

    if (left == format && right.empty()) {
      // The header was not found.  Just exit.
      break;
    }

    bool found_code = false;
    for (const auto &code : codes) {
      if (!right.consume_front(code.name))
        continue;

      if (do_color)
        fmt.append(code.value);
      found_code = true;
      break;
    }
    format = right;
    // If we haven't found a valid replacement value, we just copy the string
    // to the result without any modifications.
    if (!found_code)
      fmt.append(tok_hdr);
  }
  return fmt;
}

inline std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>
FindNextAnsiSequence(llvm::StringRef str) {
  llvm::StringRef left;
  llvm::StringRef right = str;

  while (!right.empty()) {
    const size_t start = right.find(ANSI_ESC_START);

    // ANSI_ESC_START not found.
    if (start == llvm::StringRef::npos)
      return {str, {}, {}};

    // Split the string around the current ANSI_ESC_START.
    left = str.take_front(left.size() + start);
    llvm::StringRef escape = right.substr(start);
    right = right.substr(start + ANSI_ESC_START_LEN + 1);

    const size_t end = right.find_first_not_of("0123456789;");

    // ANSI_ESC_END found.
    if (end < right.size() && (right[end] == 'm' || right[end] == 'G'))
      return {left, escape.take_front(ANSI_ESC_START_LEN + 1 + end + 1),
              right.substr(end + 1)};

    // Maintain the invariant that str == left + right at the start of the loop.
    left = str.take_front(left.size() + ANSI_ESC_START_LEN + 1);
  }

  return {str, {}, {}};
}

inline std::string StripAnsiTerminalCodes(llvm::StringRef str) {
  std::string stripped;
  while (!str.empty()) {
    auto [left, escape, right] = FindNextAnsiSequence(str);
    stripped += left;
    str = right;
  }
  return stripped;
}

inline std::string TrimAndPad(llvm::StringRef str, size_t visible_length,
                              char padding = ' ') {
  std::string result;
  result.reserve(visible_length);
  size_t result_visibile_length = 0;

  // Trim the string to the given visible length.
  while (!str.empty()) {
    auto [left, escape, right] = FindNextAnsiSequence(str);
    str = right;

    // Compute the length of the string without escape codes. If it fits, append
    // it together with the invisible escape code.
    size_t column_width = llvm::sys::locale::columnWidth(left);
    if (result_visibile_length + column_width <= visible_length) {
      result.append(left).append(escape);
      result_visibile_length += column_width;
      continue;
    }

    // The string might contain unicode which means it's not safe to truncate.
    // Repeatedly trim the string until it its valid unicode and fits.
    llvm::StringRef trimmed = left;
    while (!trimmed.empty()) {
      // This relies on columnWidth returning -2 for invalid/partial unicode
      // characters, which after conversion to size_t will be larger than the
      // visible width.
      column_width = llvm::sys::locale::columnWidth(trimmed);
      if (result_visibile_length + column_width <= visible_length) {
        result.append(trimmed);
        result_visibile_length += column_width;
        break;
      }
      trimmed = trimmed.drop_back();
    }
  }

  // Pad the string.
  if (result_visibile_length < visible_length)
    result.append(visible_length - result_visibile_length, padding);

  return result;
}

inline size_t ColumnWidth(llvm::StringRef str) {
  std::string stripped = ansi::StripAnsiTerminalCodes(str);
  return llvm::sys::locale::columnWidth(stripped);
}

struct VisibleActualPositionPair {
  size_t visible;
  size_t actual;

  bool operator==(const VisibleActualPositionPair &rhs) const {
    return visible == rhs.visible && actual == rhs.actual;
  }
};

/// This function converts a position in "visible" text (text with ANSI codes
/// removed) into a position in the data of the text (which includes ANSI
/// codes). That actual position should be used when printing out the data.
//
/// If a character is preceeded by an ANSI code, the returned position will
/// point to that ANSI code. As formatting that visible character requires the
/// code.
//
/// This logic does not extend to whole words. This function assumes you are not
/// going to split up the words of whatever text you pass in here. Making sure
/// not to split words is the responsibility of the caller of this function.
//
/// Returns a pair containing {visible position, actual position}. This may be
/// passed back to the function later as "hint" to allow it to skip ahead if you
/// are asking for a visible position equal or greater to the one in the hint.
//
/// The function assumes that the hint provided is correct, even if it refers
/// to a position that does not exist. If the hint is for the exact requested
/// visible position, it will return the hint, without checking the string at
/// all.
///
/// FIXME: This function is not Unicode aware.
inline VisibleActualPositionPair
VisiblePositionToActualPosition(llvm::StringRef text, size_t visible_position,
                                std::optional<VisibleActualPositionPair> hint) {
  size_t actual_position = 0;
  const size_t wanted_visible_position = visible_position;
  visible_position = 0;
  llvm::StringRef remaining_text = text;

  if (hint) {
    if (hint->visible == wanted_visible_position)
      return *hint;

    if (hint->visible < wanted_visible_position) {
      // Skip forward using the hint.
      visible_position = hint->visible;
      actual_position = hint->actual;

      remaining_text = remaining_text.drop_front(actual_position);
    }
  }

  while (remaining_text.size()) {
    auto [left, escape, right] = ansi::FindNextAnsiSequence(remaining_text);

    // FIXME: We are assuming left.size() ==  the number of visible characters
    // on the left. This is not true for Unicode.
    for (unsigned i = 0; i < left.size(); ++i) {
      if (visible_position == wanted_visible_position)
        return {wanted_visible_position, actual_position};

      actual_position++;
      visible_position++;
    }

    if (visible_position == wanted_visible_position)
      return {wanted_visible_position, actual_position};

    actual_position += escape.size();
    remaining_text = right;
  }

  assert(visible_position == wanted_visible_position &&
         "should have found visible_position by now");

  return {wanted_visible_position, actual_position};
}

// Output text that may contain ANSI codes, word wrapped (wrapped at whitespace)
// to the given stream. The indent level of the stream is counted towards the
// output line length.
// FIXME: This does not handle unicode correctly.
// FIXME: If an ANSI code is applied to multiple words and those words are split
//        across lines, the code will apply to the indentation as well as the
//        text.
inline void OutputWordWrappedLines(Stream &strm, llvm::StringRef text,
                                   uint32_t output_max_columns) {
  // We will indent using the stream, so leading whitespace is not significant.
  text = text.ltrim();
  if (text.size() == 0)
    return;

  const size_t visible_length = ansi::ColumnWidth(text);

  // Will it all fit on one line, or is it a single word that we must not break?
  if (static_cast<uint32_t>(visible_length + strm.GetIndentLevel()) <
          output_max_columns ||
      text.find_first_of(" \t\n") == llvm::StringRef::npos) {
    // Output it as a single line.
    strm.Indent(text);
    strm.EOL();
    return;
  }

  // We need to break it up into multiple lines. We can do this based on the
  // formatted text because we know that:
  // * We only break lines on whitespace, therefore we will not break in the
  //   middle of a Unicode character or escape code.
  // * Escape codes are so far not applied to multiple words, so there is no
  //   risk of breaking up a phrase and the escape code being incorrectly
  //   applied to the indent too.

  const int max_text_width = output_max_columns - strm.GetIndentLevel() - 1;
  // start, end and final_end are all positions in the visible text,
  // not the data representing that text.
  int start = 0;
  int end = start;
  const int final_end = visible_length;
  std::optional<ansi::VisibleActualPositionPair> conversion_hint;

  // FIXME: This removes ANSI but unicode will still take up > 1 byte per
  // character.
  // We can either constantly convert between visible and actual position,
  // or make a copy and only convert at the end. Assume that a copy is cheap to
  // do.
  const std::string text_without_ansi = ansi::StripAnsiTerminalCodes(text);

  while (end < final_end) {
    // Don't start the 'text' on a space, since we're already outputting the
    // indentation.
    while ((start < final_end) && (text_without_ansi[start] == ' '))
      start++;

    end = start + max_text_width;
    if (end > final_end)
      end = final_end;

    if (end != final_end) {
      // If we're not at the end of the text, make sure we break the line on
      // white space.
      while (end > start && text_without_ansi[end] != ' ' &&
             text_without_ansi[end] != '\t' && text_without_ansi[end] != '\n')
        end--;
    }

    if (start != 0)
      strm.EOL();
    strm.Indent();

    const int sub_len = end - start;
    UNUSED_IF_ASSERT_DISABLED(sub_len);
    assert(start < final_end);
    assert(start + sub_len <= final_end);

    conversion_hint =
        ansi::VisiblePositionToActualPosition(text, start, conversion_hint);
    const size_t start_actual = conversion_hint->actual;
    conversion_hint =
        ansi::VisiblePositionToActualPosition(text, end, conversion_hint);
    size_t end_actual = conversion_hint->actual;

    // If the end is proceeded by an ANSI code, include that code.
    llvm::StringRef end_of_line = text.substr(end_actual);
    auto [left, escape, right] = ansi::FindNextAnsiSequence(end_of_line);
    if (left.size() == 0)
      end_actual += escape.size();

    const int sub_len_actual = end_actual - start_actual;
    strm << text.substr(start_actual, sub_len_actual);
    start = end + 1;
  }
  strm.EOL();
}

} // namespace ansi
} // namespace lldb_private

#endif
