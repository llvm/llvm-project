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
#include "main.h"
#include "flang/Common/format.h"
#include "flang/Decimal/decimal.h"
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
int FormatControl<CONTEXT>::GetMaxParenthesisNesting(
    IoErrorHandler &handler, const CharType *format, std::size_t formatLength) {
  int maxNesting{0};
  int nesting{0};
  const CharType *end{format + formatLength};
  std::optional<CharType> quote;
  int repeat{0};
  for (const CharType *p{format}; p < end; ++p) {
    if (quote) {
      if (*p == *quote) {
        quote.reset();
      }
    } else if (*p >= '0' && *p <= '9') {
      repeat = 10 * repeat + *p - '0';
    } else if (*p != ' ') {
      switch (*p) {
      case '\'':
      case '"':
        quote = *p;
        break;
      case 'h':
      case 'H': // 9HHOLLERITH
        p += repeat;
        if (p >= end) {
          handler.SignalError(IostatErrorInFormat,
              "Hollerith (%dH) too long in FORMAT", repeat);
          return maxNesting;
        }
        break;
      case ' ':
        break;
      case '(':
        ++nesting;
        maxNesting = std::max(nesting, maxNesting);
        break;
      case ')':
        nesting = std::max(nesting - 1, 0);
        break;
      }
      repeat = 0;
    }
  }
  if (quote) {
    handler.SignalError(
        IostatErrorInFormat, "Unbalanced quotation marks in FORMAT string");
  } else if (nesting) {
    handler.SignalError(
        IostatErrorInFormat, "Unbalanced parentheses in FORMAT string");
  }
  return maxNesting;
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
    firstCh = '\0';
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
        context.SignalError(IostatErrorInFormat,
            "Invalid FORMAT: '*' may appear only before '('");
        return 0;
      }
    }
    ch = Capitalize(ch);
    if (ch == '(') {
      if (height_ >= maxHeight_) {
        context.SignalError(IostatErrorInFormat,
            "FORMAT stack overflow: too many nested parentheses");
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
        stack_[0].start = maybeReversionPoint - 1;
      }
      ++height_;
    } else if (height_ == 0) {
      context.SignalError(IostatErrorInFormat, "FORMAT lacks initial '('");
      return 0;
    } else if (ch == ')') {
      if (height_ == 1) {
        if (stop) {
          return 0; // end of FORMAT and no data items remain
        }
        context.AdvanceRecord(); // implied / before rightmost )
      }
      auto restart{stack_[height_ - 1].start + 1};
      if (stack_[height_ - 1].remaining == Iteration::unlimited) {
        offset_ = restart;
        if (offset_ == unlimitedLoopCheck) {
          context.SignalError(IostatErrorInFormat,
              "Unlimited repetition in FORMAT lacks data edit descriptors");
        }
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
        context.SignalError(IostatErrorInFormat,
            "FORMAT missing closing quote on character literal");
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
        context.SignalError(
            IostatErrorInFormat, "Invalid width on Hollerith in FORMAT");
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
          next = peek;
          ++offset_;
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
    } else {
      context.SignalError(IostatErrorInFormat,
          "Invalid character '%c' in FORMAT", static_cast<char>(ch));
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
    // DT'iotype'(v_list) user-defined derived type I/O
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
        context.SignalError(
            IostatErrorInFormat, "Unclosed DT'iotype' in FORMAT");
      } else if (tooLong) {
        context.SignalError(
            IostatErrorInFormat, "Excessive DT'iotype' in FORMAT");
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
        context.SignalError(
            IostatErrorInFormat, "Unclosed DT(v_list) in FORMAT");
      } else if (tooLong) {
        context.SignalError(
            IostatErrorInFormat, "Excessive DT(v_list) in FORMAT");
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
  if (repeat > maxRepeat) {
    stack_[height_].start = start; // after repeat count
    stack_[height_].remaining = repeat; // full count
    ++height_;
  }
  edit.repeat = std::min(1, maxRepeat); // 0 if maxRepeat==0
  if (height_ > 1) { // Subtle: stack_[0].start doesn't necessarily point to '('
    int start{stack_[height_ - 1].start};
    if (format_[start] != '(') {
      if (stack_[height_ - 1].remaining > maxRepeat) {
        edit.repeat = maxRepeat;
        stack_[height_ - 1].remaining -= maxRepeat;
        offset_ = start; // repeat same edit descriptor next time
      } else {
        edit.repeat = stack_[height_ - 1].remaining;
        --height_;
      }
    }
  }
  return edit;
}

template <typename CONTEXT>
void FormatControl<CONTEXT>::Finish(Context &context) {
  CueUpNextDataEdit(context, true /* stop at colon or end of FORMAT */);
}
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_FORMAT_IMPLEMENTATION_H_
