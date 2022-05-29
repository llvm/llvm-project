//===-- runtime/format-implementation.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements out-of-line member functions of template class FormatControl

#ifndef FORTRAN_RUNTIME_FORMAT_IMPLEMENTATION_H_
#define FORTRAN_RUNTIME_FORMAT_IMPLEMENTATION_H_

#include "format.h"
#include "io-stmt.h"
#include "flang/Common/format.h"
#include "flang/Decimal/decimal.h"
#include "flang/Runtime/main.h"
#include <algorithm>
#include <limits>

namespace Fortran::runtime::io {

template <typename CONTEXT>
FormatControl<CONTEXT>::FormatControl(const Terminator &terminator,
    const CharType *format, std::size_t formatLength, int maxHeight)
    : maxHeight_{static_cast<std::uint8_t>(maxHeight)}, format_{format},
      formatLength_{static_cast<int>(formatLength)} {
  RUNTIME_CHECK(terminator, maxHeight == maxHeight_);
  RUNTIME_CHECK(
      terminator, formatLength == static_cast<std::size_t>(formatLength_));
  stack_[0].start = offset_;
  stack_[0].remaining = Iteration::unlimited; // 13.4(8)
}

template <typename CONTEXT>
int FormatControl<CONTEXT>::GetIntField(
    IoErrorHandler &handler, CharType firstCh) {
  CharType ch{firstCh ? firstCh : PeekNext()};
  if (ch != '-' && ch != '+' && (ch < '0' || ch > '9')) {
    handler.SignalError(IostatErrorInFormat,
        "Invalid FORMAT: integer expected at '%c'", static_cast<char>(ch));
    return 0;
  }
  int result{0};
  bool negate{ch == '-'};
  if (negate || ch == '+') {
    if (firstCh) {
      firstCh = '\0';
    } else {
      ++offset_;
    }
    ch = PeekNext();
  }
  while (ch >= '0' && ch <= '9') {
    if (result >
        std::numeric_limits<int>::max() / 10 - (static_cast<int>(ch) - '0')) {
      handler.SignalError(
          IostatErrorInFormat, "FORMAT integer field out of range");
      return result;
    }
    result = 10 * result + ch - '0';
    if (firstCh) {
      firstCh = '\0';
    } else {
      ++offset_;
    }
    ch = PeekNext();
  }
  if (negate && (result *= -1) > 0) {
    handler.SignalError(
        IostatErrorInFormat, "FORMAT integer field out of range");
  }
  return result;
}

template <typename CONTEXT>
static void HandleControl(CONTEXT &context, char ch, char next, int n) {
  MutableModes &modes{context.mutableModes()};
  switch (ch) {
  case 'B':
    if (next == 'Z') {
      modes.editingFlags |= blankZero;
      return;
    }
    if (next == 'N') {
      modes.editingFlags &= ~blankZero;
      return;
    }
    break;
  case 'D':
    if (next == 'C') {
      modes.editingFlags |= decimalComma;
      return;
    }
    if (next == 'P') {
      modes.editingFlags &= ~decimalComma;
      return;
    }
    break;
  case 'P':
    if (!next) {
      modes.scale = n; // kP - decimal scaling by 10**k
      return;
    }
    break;
  case 'R':
    switch (next) {
    case 'N':
      modes.round = decimal::RoundNearest;
      return;
    case 'Z':
      modes.round = decimal::RoundToZero;
      return;
    case 'U':
      modes.round = decimal::RoundUp;
      return;
    case 'D':
      modes.round = decimal::RoundDown;
      return;
    case 'C':
      modes.round = decimal::RoundCompatible;
      return;
    case 'P':
      modes.round = executionEnvironment.defaultOutputRoundingMode;
      return;
    default:
      break;
    }
    break;
  case 'X':
    if (!next) {
      context.HandleRelativePosition(n);
      return;
    }
    break;
  case 'S':
    if (next == 'P') {
      modes.editingFlags |= signPlus;
      return;
    }
    if (!next || next == 'S') {
      modes.editingFlags &= ~signPlus;
      return;
    }
    break;
  case 'T': {
    if (!next) { // Tn
      context.HandleAbsolutePosition(n - 1); // convert 1-based to 0-based
      return;
    }
    if (next == 'L' || next == 'R') { // TLn & TRn
      context.HandleRelativePosition(next == 'L' ? -n : n);
      return;
    }
  } break;
  default:
    break;
  }
  if (next) {
    context.SignalError(IostatErrorInFormat,
        "Unknown '%c%c' edit descriptor in FORMAT", ch, next);
  } else {
    context.SignalError(
        IostatErrorInFormat, "Unknown '%c' edit descriptor in FORMAT", ch);
  }
}

// Locates the next data edit descriptor in the format.
// Handles all repetition counts and control edit descriptors.
// Generally assumes that the format string has survived the common
// format validator gauntlet.
template <typename CONTEXT>
int FormatControl<CONTEXT>::CueUpNextDataEdit(Context &context, bool stop) {
  int unlimitedLoopCheck{-1};
  // Do repetitions remain on an unparenthesized data edit?
  while (height_ > 1 && format_[stack_[height_ - 1].start] != '(') {
    offset_ = stack_[height_ - 1].start;
    int repeat{stack_[height_ - 1].remaining};
    --height_;
    if (repeat > 0) {
      return repeat;
    }
  }
  while (true) {
    std::optional<int> repeat;
    bool unlimited{false};
    auto maybeReversionPoint{offset_};
    CharType ch{GetNextChar(context)};
    while (ch == ',' || ch == ':') {
      // Skip commas, and don't complain if they're missing; the format
      // validator does that.
      if (stop && ch == ':') {
        return 0;
      }
      ch = GetNextChar(context);
    }
    if (ch == '-' || ch == '+' || (ch >= '0' && ch <= '9')) {
      repeat = GetIntField(context, ch);
      ch = GetNextChar(context);
    } else if (ch == '*') {
      unlimited = true;
      ch = GetNextChar(context);
      if (ch != '(') {
        ReportBadFormat(context,
            "Invalid FORMAT: '*' may appear only before '('",
            maybeReversionPoint);
        return 0;
      }
      if (height_ != 1) {
        ReportBadFormat(context,
            "Invalid FORMAT: '*' must be nested in exactly one set of "
            "parentheses",
            maybeReversionPoint);
        return 0;
      }
    }
    ch = Capitalize(ch);
    if (ch == '(') {
      if (height_ >= maxHeight_) {
        ReportBadFormat(context,
            "FORMAT stack overflow: too many nested parentheses",
            maybeReversionPoint);
        return 0;
      }
      stack_[height_].start = offset_ - 1; // the '('
      RUNTIME_CHECK(context, format_[stack_[height_].start] == '(');
      if (unlimited || height_ == 0) {
        stack_[height_].remaining = Iteration::unlimited;
        unlimitedLoopCheck = offset_ - 1;
      } else if (repeat) {
        if (*repeat <= 0) {
          *repeat = 1; // error recovery
        }
        stack_[height_].remaining = *repeat - 1;
      } else {
        stack_[height_].remaining = 0;
      }
      if (height_ == 1) {
        // Subtle point (F'2018 13.4 para 9): tha last parenthesized group
        // at height 1 becomes the restart point after control reaches the
        // end of the format, including its repeat count.
        stack_[0].start = maybeReversionPoint;
      }
      ++height_;
    } else if (height_ == 0) {
      ReportBadFormat(context, "FORMAT lacks initial '('", maybeReversionPoint);
      return 0;
    } else if (ch == ')') {
      if (height_ == 1) {
        if (stop) {
          return 0; // end of FORMAT and no data items remain
        }
        context.AdvanceRecord(); // implied / before rightmost )
      }
      auto restart{stack_[height_ - 1].start};
      if (format_[restart] == '(') {
        ++restart;
      }
      if (stack_[height_ - 1].remaining == Iteration::unlimited) {
        if (height_ > 1 && GetNextChar(context) != ')') {
          ReportBadFormat(context,
              "Unlimited repetition in FORMAT may not be followed by more "
              "items",
              restart);
          return 0;
        }
        if (offset_ == unlimitedLoopCheck) {
          ReportBadFormat(context,
              "Unlimited repetition in FORMAT lacks data edit descriptors",
              restart);
          return 0;
        }
        offset_ = restart;
      } else if (stack_[height_ - 1].remaining-- > 0) {
        offset_ = restart;
      } else {
        --height_;
      }
    } else if (ch == '\'' || ch == '"') {
      // Quoted 'character literal'
      CharType quote{ch};
      auto start{offset_};
      while (offset_ < formatLength_ && format_[offset_] != quote) {
        ++offset_;
      }
      if (offset_ >= formatLength_) {
        ReportBadFormat(context,
            "FORMAT missing closing quote on character literal",
            maybeReversionPoint);
        return 0;
      }
      ++offset_;
      std::size_t chars{
          static_cast<std::size_t>(&format_[offset_] - &format_[start])};
      if (PeekNext() == quote) {
        // subtle: handle doubled quote character in a literal by including
        // the first in the output, then treating the second as the start
        // of another character literal.
      } else {
        --chars;
      }
      context.Emit(format_ + start, chars);
    } else if (ch == 'H') {
      // 9HHOLLERITH
      if (!repeat || *repeat < 1 || offset_ + *repeat > formatLength_) {
        ReportBadFormat(context, "Invalid width on Hollerith in FORMAT",
            maybeReversionPoint);
        return 0;
      }
      context.Emit(format_ + offset_, static_cast<std::size_t>(*repeat));
      offset_ += *repeat;
    } else if (ch >= 'A' && ch <= 'Z') {
      int start{offset_ - 1};
      CharType next{'\0'};
      if (ch != 'P') { // 1PE5.2 - comma not required (C1302)
        CharType peek{Capitalize(PeekNext())};
        if (peek >= 'A' && peek <= 'Z') {
          if (ch == 'A' /* anticipate F'202X AT editing */ || ch == 'B' ||
              ch == 'D' || ch == 'E' || ch == 'R' || ch == 'S' || ch == 'T') {
            // Assume a two-letter edit descriptor
            next = peek;
            ++offset_;
          } else {
            // extension: assume a comma between 'ch' and 'peek'
          }
        }
      }
      if ((!next &&
              (ch == 'A' || ch == 'I' || ch == 'B' || ch == 'E' || ch == 'D' ||
                  ch == 'O' || ch == 'Z' || ch == 'F' || ch == 'G' ||
                  ch == 'L')) ||
          (ch == 'E' && (next == 'N' || next == 'S' || next == 'X')) ||
          (ch == 'D' && next == 'T')) {
        // Data edit descriptor found
        offset_ = start;
        return repeat && *repeat > 0 ? *repeat : 1;
      } else {
        // Control edit descriptor
        if (ch == 'T') { // Tn, TLn, TRn
          repeat = GetIntField(context);
        }
        HandleControl(context, static_cast<char>(ch), static_cast<char>(next),
            repeat ? *repeat : 1);
      }
    } else if (ch == '/') {
      context.AdvanceRecord(repeat && *repeat > 0 ? *repeat : 1);
    } else if (ch == '$' || ch == '\\') {
      context.mutableModes().nonAdvancing = true;
    } else if (ch == '\t' || ch == '\v') {
      // Tabs (extension)
      // TODO: any other raw characters?
      context.Emit(format_ + offset_ - 1, 1);
    } else {
      ReportBadFormat(
          context, "Invalid character in FORMAT", maybeReversionPoint);
      return 0;
    }
  }
}

// Returns the next data edit descriptor
template <typename CONTEXT>
DataEdit FormatControl<CONTEXT>::GetNextDataEdit(
    Context &context, int maxRepeat) {
  int repeat{CueUpNextDataEdit(context)};
  auto start{offset_};
  DataEdit edit;
  edit.descriptor = static_cast<char>(Capitalize(GetNextChar(context)));
  if (edit.descriptor == 'E') {
    if (auto next{static_cast<char>(Capitalize(PeekNext()))};
        next == 'N' || next == 'S' || next == 'X') {
      edit.variation = next;
      ++offset_;
    }
  } else if (edit.descriptor == 'D' && Capitalize(PeekNext()) == 'T') {
    // DT['iotype'][(v_list)] user-defined derived type I/O
    edit.descriptor = DataEdit::DefinedDerivedType;
    ++offset_;
    if (auto quote{static_cast<char>(PeekNext())};
        quote == '\'' || quote == '"') {
      // Capture the quoted 'iotype'
      bool ok{false}, tooLong{false};
      for (++offset_; offset_ < formatLength_;) {
        auto ch{static_cast<char>(format_[offset_++])};
        if (ch == quote &&
            (offset_ == formatLength_ ||
                static_cast<char>(format_[offset_]) != quote)) {
          ok = true;
          break; // that was terminating quote
        } else if (edit.ioTypeChars >= edit.maxIoTypeChars) {
          tooLong = true;
        } else {
          edit.ioType[edit.ioTypeChars++] = ch;
          if (ch == quote) {
            ++offset_;
          }
        }
      }
      if (!ok) {
        ReportBadFormat(context, "Unclosed DT'iotype' in FORMAT", start);
      } else if (tooLong) {
        ReportBadFormat(context, "Excessive DT'iotype' in FORMAT", start);
      }
    }
    if (PeekNext() == '(') {
      // Capture the v_list arguments
      bool ok{false}, tooLong{false};
      for (++offset_; offset_ < formatLength_;) {
        int n{GetIntField(context)};
        if (edit.vListEntries >= edit.maxVListEntries) {
          tooLong = true;
        } else {
          edit.vList[edit.vListEntries++] = n;
        }
        auto ch{static_cast<char>(GetNextChar(context))};
        if (ch != ',') {
          ok = ch == ')';
          break;
        }
      }
      if (!ok) {
        ReportBadFormat(context, "Unclosed DT(v_list) in FORMAT", start);
      } else if (tooLong) {
        ReportBadFormat(context, "Excessive DT(v_list) in FORMAT", start);
      }
    }
  }
  if (edit.descriptor == 'A') { // width is optional for A[w]
    auto ch{PeekNext()};
    if (ch >= '0' && ch <= '9') {
      edit.width = GetIntField(context);
    }
  } else if (edit.descriptor != DataEdit::DefinedDerivedType) {
    edit.width = GetIntField(context);
  }
  if constexpr (std::is_base_of_v<InputStatementState, CONTEXT>) {
    if (edit.width.value_or(-1) == 0) {
      ReportBadFormat(context, "Input field width is zero", start);
    }
  }
  if (edit.descriptor != DataEdit::DefinedDerivedType && PeekNext() == '.') {
    ++offset_;
    edit.digits = GetIntField(context);
    CharType ch{PeekNext()};
    if (ch == 'e' || ch == 'E' || ch == 'd' || ch == 'D') {
      ++offset_;
      edit.expoDigits = GetIntField(context);
    }
  }
  edit.modes = context.mutableModes();
  // Handle repeated nonparenthesized edit descriptors
  edit.repeat = std::min(repeat, maxRepeat); // 0 if maxRepeat==0
  if (repeat > maxRepeat) {
    stack_[height_].start = start; // after repeat count
    stack_[height_].remaining = repeat - edit.repeat;
    ++height_;
  }
  return edit;
}

template <typename CONTEXT>
void FormatControl<CONTEXT>::Finish(Context &context) {
  CueUpNextDataEdit(context, true /* stop at colon or end of FORMAT */);
}
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_FORMAT_IMPLEMENTATION_H_
