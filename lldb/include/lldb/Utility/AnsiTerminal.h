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
#include "llvm/Support/Unicode.h"

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

inline size_t ColumnWidth(llvm::StringRef str) {
  std::string stripped = ansi::StripAnsiTerminalCodes(str);
  return llvm::sys::locale::columnWidth(stripped);
}

/// Trim the given string to the given visible length, at a word boundary.
/// Visible length means its width when rendered to the terminal.
/// The string can include ANSI codes and Unicode.
///
/// For a single word string, that word is returned in its entirety regardless
/// of its visible length.
///
/// This function is similar to TrimAndPad, except that it must split on a word
/// boundary. So there are some notable differences:
/// * Has a special case for single words that exceed desired visible
///   length.
/// * Must track whether the most recent modifications was on a word boundary
///   or not.
/// * If the trimming finishes without the result ending on a word boundary,
///   it must find the nearest boundary to that trim point by trimming more.
inline std::string TrimAtWordBoundary(llvm::StringRef str,
                                      size_t visible_length) {
  str = str.trim();
  if (str.empty())
    return str.str();

  auto first_whitespace = str.find_first_of(" \t\n");
  // No whitespace means a single word, which we cannot split.
  if (first_whitespace == llvm::StringRef::npos)
    return str.str();

  // If the first word of a multi-word string is too wide, return that whole
  // word only.
  auto to_first_word_boundary = str.substr(0, first_whitespace);
  // We use ansi::ColumnWidth here because it can handle ANSI and Unicode.
  if (ansi::ColumnWidth(to_first_word_boundary) > visible_length)
    return to_first_word_boundary.str();

  std::string result;
  result.reserve(visible_length);
  // When there is Unicode or ANSI codes, the visible length will not equal
  // result.size(), so we track it separately.
  size_t result_visible_length = 0;

  // The loop below makes many adjustments, and we never know which will be the
  // last. This tracks whether the most recent adjustment put us at a word
  // boundary and is checked after the main loop.
  bool at_word_boundary = false;

  // Trim the string to the given visible length.
  while (!str.empty()) {
    auto [left, escape, right] = FindNextAnsiSequence(str);
    str = right;

    // We know that left does not include ANSI codes. Compute its visible length
    // and if it fits, append it together with the invisible escape code.
    size_t column_width = llvm::sys::locale::columnWidth(left);
    if (result_visible_length + column_width <= visible_length) {
      result.append(left).append(escape);
      result_visible_length += column_width;
      at_word_boundary = right.empty() || std::isspace(right[0]);

      continue;
    }

    // The string might contain unicode which means it's not safe to truncate.
    // Repeatedly trim the string until it is valid unicode and fits.
    llvm::StringRef trimmed = left;

    // A word break can happen at the character we trim to, or the one we
    // trimmed before that (we are going backwards, so before in the loop is
    // after in the string).

    // A word break can happen at the point we trim, or just beyond that point.
    // In other words: at the current back of trimmed, or what was the back last
    // time around. following_char records the character popped in the previous
    // loop iteration.
    std::optional<char> following_char = std::nullopt;
    while (!trimmed.empty()) {
      int trimmed_width = llvm::sys::locale::columnWidth(trimmed);
      if (
          // If we have a partial Unicode character, keep trimming.
          trimmed_width !=
              llvm::sys::unicode::ColumnWidthErrors::ErrorInvalidUTF8 &&
          // If the trimmed string fits in the column limit, stop trimming.
          (result_visible_length + static_cast<size_t>(trimmed_width) <=
           visible_length)) {
        result.append(trimmed);
        result_visible_length += trimmed_width;
        at_word_boundary = std::isspace(trimmed.back()) ||
                           (following_char && std::isspace(*following_char));

        break;
      }

      following_char = trimmed.back();
      trimmed = trimmed.drop_back();
    }
  }

  if (!at_word_boundary) {
    // Walk backwards to find a word boundary.
    auto last_whitespace = result.find_last_of(" \t\n");
    if (last_whitespace != std::string::npos)
      result = result.substr(0, last_whitespace);
  }

  // We may have split on whitespace that was the first of a word boundary, or
  // somewhere in a run of whitespace. Trim the trailing spaces. This must be
  // done here instead of in the loop because in the loop we may still be
  // accumulating the result string.
  return llvm::StringRef(result).rtrim().str();
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
      int trimmed_width = llvm::sys::locale::columnWidth(trimmed);
      if (
          // If we have only part of a Unicode character, keep trimming.
          trimmed_width !=
              llvm::sys::unicode::ColumnWidthErrors::ErrorInvalidUTF8 &&
          // If the trimmed string fits, take it.
          result_visibile_length + static_cast<size_t>(trimmed_width) <=
              visible_length) {
        result.append(trimmed);
        result_visibile_length += static_cast<size_t>(trimmed_width);
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

// Output text that may contain ANSI codes, word wrapped (wrapped at whitespace)
// to the given stream. The indent level of the stream is counted towards the
// output line length.
// FIXME: If an ANSI code is applied to multiple words and those words are split
//        across lines, the code will apply to the indentation as well as the
//        text.
inline void OutputWordWrappedLines(Stream &strm, llvm::StringRef text,
                                   uint32_t output_max_columns) {
  // We will indent using the stream, so leading whitespace is not significant.
  text = text.ltrim();
  if (text.empty())
    return;

  // 1 column border on the right side.
  const uint32_t max_text_width =
      output_max_columns - strm.GetIndentLevel() - 1;
  bool first_line = true;

  while (!text.empty()) {
    std::string split = TrimAtWordBoundary(text, max_text_width);
    if (!first_line)
      strm.EOL();
    first_line = false;
    strm.Indent(split);

    text = text.drop_front(split.size()).ltrim();
  }

  strm.EOL();
}

} // namespace ansi
} // namespace lldb_private

#endif
