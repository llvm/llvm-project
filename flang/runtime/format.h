//===-- runtime/format.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FORMAT string processing

#ifndef FORTRAN_RUNTIME_FORMAT_H_
#define FORTRAN_RUNTIME_FORMAT_H_

#include "environment.h"
#include "io-error.h"
#include "flang/Common/Fortran.h"
#include "flang/Decimal/decimal.h"
#include <cinttypes>
#include <optional>

namespace Fortran::runtime::io {

enum EditingFlags {
  blankZero = 1, // BLANK=ZERO or BZ edit
  decimalComma = 2, // DECIMAL=COMMA or DC edit
  signPlus = 4, // SIGN=PLUS or SP edit
};

struct MutableModes {
  std::uint8_t editingFlags{0}; // BN, DP, SS
  enum decimal::FortranRounding round{
      executionEnvironment
          .defaultOutputRoundingMode}; // RP/ROUND='PROCESSOR_DEFAULT'
  bool pad{true}; // PAD= mode on READ
  char delim{'\0'}; // DELIM=
  short scale{0}; // kP
  bool inNamelist{false}; // skip ! comments
  bool nonAdvancing{false}; // ADVANCE='NO', or $ or \ in FORMAT
};

// A single edit descriptor extracted from a FORMAT
struct DataEdit {
  char descriptor; // capitalized: one of A, I, B, O, Z, F, E(N/S/X), D, G

  // Special internal data edit descriptors for list-directed & NAMELIST I/O
  static constexpr char ListDirected{'g'}; // non-COMPLEX list-directed
  static constexpr char ListDirectedRealPart{'r'}; // emit "(r," or "(r;"
  static constexpr char ListDirectedImaginaryPart{'z'}; // emit "z)"
  static constexpr char ListDirectedNullValue{'n'}; // see 13.10.3.2
  constexpr bool IsListDirected() const {
    return descriptor == ListDirected || descriptor == ListDirectedRealPart ||
        descriptor == ListDirectedImaginaryPart;
  }
  constexpr bool IsNamelist() const {
    return IsListDirected() && modes.inNamelist;
  }

  static constexpr char DefinedDerivedType{'d'}; // DT user-defined derived type

  char variation{'\0'}; // N, S, or X for EN, ES, EX
  std::optional<int> width; // the 'w' field; optional for A
  std::optional<int> digits; // the 'm' or 'd' field
  std::optional<int> expoDigits; // 'Ee' field
  MutableModes modes;
  int repeat{1};

  // "iotype" &/or "v_list" values for a DT'iotype'(v_list)
  // user-defined derived type data edit descriptor
  static constexpr std::size_t maxIoTypeChars{32};
  static constexpr std::size_t maxVListEntries{4};
  std::uint8_t ioTypeChars{0};
  std::uint8_t vListEntries{0};
  char ioType[maxIoTypeChars];
  int vList[maxVListEntries];
};

// Generates a sequence of DataEdits from a FORMAT statement or
// default-CHARACTER string.  Driven by I/O item list processing.
// Errors are fatal.  See subclause 13.4 in Fortran 2018 for background.
template <typename CONTEXT> class FormatControl {
public:
  using Context = CONTEXT;
  using CharType = typename Context::CharType;

  FormatControl() {}
  FormatControl(const Terminator &, const CharType *format,
      std::size_t formatLength, int maxHeight = maxMaxHeight);

  // Determines the max parenthesis nesting level by scanning and validating
  // the FORMAT string.
  static int GetMaxParenthesisNesting(
      IoErrorHandler &, const CharType *format, std::size_t formatLength);

  // For attempting to allocate in a user-supplied stack area
  static std::size_t GetNeededSize(int maxHeight) {
    return sizeof(FormatControl) -
        sizeof(Iteration) * (maxMaxHeight - maxHeight);
  }

  // Extracts the next data edit descriptor, handling control edit descriptors
  // along the way.  If maxRepeat==0, this is a peek at the next data edit
  // descriptor.
  DataEdit GetNextDataEdit(Context &, int maxRepeat = 1);

  // Emit any remaining character literals after the last data item (on output)
  // and perform remaining record positioning actions.
  void Finish(Context &);

private:
  static constexpr std::uint8_t maxMaxHeight{100};

  struct Iteration {
    static constexpr int unlimited{-1};
    int start{0}; // offset in format_ of '(' or a repeated edit descriptor
    int remaining{0}; // while >0, decrement and iterate
  };

  void SkipBlanks() {
    while (offset_ < formatLength_ && format_[offset_] == ' ') {
      ++offset_;
    }
  }
  CharType PeekNext() {
    SkipBlanks();
    return offset_ < formatLength_ ? format_[offset_] : '\0';
  }
  CharType GetNextChar(IoErrorHandler &handler) {
    SkipBlanks();
    if (offset_ >= formatLength_) {
      handler.SignalError(
          IostatErrorInFormat, "FORMAT missing at least one ')'");
      return '\n';
    }
    return format_[offset_++];
  }
  int GetIntField(IoErrorHandler &, CharType firstCh = '\0');

  // Advances through the FORMAT until the next data edit
  // descriptor has been found; handles control edit descriptors
  // along the way.  Returns the repeat count that appeared
  // before the descriptor (defaulting to 1) and leaves offset_
  // pointing to the data edit.
  int CueUpNextDataEdit(Context &, bool stop = false);

  static constexpr CharType Capitalize(CharType ch) {
    return ch >= 'a' && ch <= 'z' ? ch + 'A' - 'a' : ch;
  }

  // Data members are arranged and typed so as to reduce size.
  // This structure may be allocated in stack space loaned by the
  // user program for internal I/O.
  const std::uint8_t maxHeight_{maxMaxHeight};
  std::uint8_t height_{0};
  const CharType *format_{nullptr};
  int formatLength_{0};
  int offset_{0}; // next item is at format_[offset_]

  // must be last, may be incomplete
  Iteration stack_[maxMaxHeight];
};
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_FORMAT_H_
