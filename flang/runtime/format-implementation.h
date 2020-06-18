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
  if (negate) {
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
      if (stack_[height_ - 1].remaining == Iteration::unlimited) {
        offset_ = stack_[height_ - 1].start + 1;
        if (offset_ == unlimitedLoopCheck) {
          context.SignalError(IostatErrorInFormat,
              "Unlimited repetition in FORMAT lacks data edit descriptors");
        }
      } else if (stack_[height_ - 1].remaining-- > 0) {
        offset_ = stack_[height_ - 1].start + 1;
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
      CharType next{Capitalize(PeekNext())};
      if (next >= 'A' && next <= 'Z') {
        ++offset_;
      } else {
        next = '\0';
      }
      if (ch == 'E' ||
          (!next &&
              (ch == 'A' || ch == 'I' || ch == 'B' || ch == 'O' || ch == 'Z' ||
                  ch == 'F' || ch == 'D' || ch == 'G' || ch == 'L'))) {
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
    } else {
      context.SignalError(IostatErrorInFormat,
          "Invalid character '%c' in FORMAT", static_cast<char>(ch));
      return 0;
    }
  }
}

template <typename CONTEXT>
DataEdit FormatControl<CONTEXT>::GetNextDataEdit(
    Context &context, int maxRepeat) {

  // TODO: DT editing

  // Return the next data edit descriptor
  int repeat{CueUpNextDataEdit(context)};
  auto start{offset_};
  DataEdit edit;
  edit.descriptor = static_cast<char>(Capitalize(GetNextChar(context)));
  if (edit.descriptor == 'E') {
    edit.variation = static_cast<char>(Capitalize(PeekNext()));
    if (edit.variation >= 'A' && edit.variation <= 'Z') {
      ++offset_;
    }
  }

  if (edit.descriptor == 'A') { // width is optional for A[w]
    auto ch{PeekNext()};
    if (ch >= '0' && ch <= '9') {
      edit.width = GetIntField(context);
    }
  } else {
    edit.width = GetIntField(context);
  }
  edit.modes = context.mutableModes();
  if (PeekNext() == '.') {
    ++offset_;
    edit.digits = GetIntField(context);
    CharType ch{PeekNext()};
    if (ch == 'e' || ch == 'E' || ch == 'd' || ch == 'D') {
      ++offset_;
      edit.expoDigits = GetIntField(context);
    }
  }

  // Handle repeated nonparenthesized edit descriptors
  if (repeat > 1) {
    stack_[height_].start = start; // after repeat count
    stack_[height_].remaining = repeat; // full count
    ++height_;
  }
  edit.repeat = 1;
  if (height_ > 1) {
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
