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
#include "flang/Common/Fortran-consts.h"
#include "flang/Common/optional.h"
#include "flang/Decimal/decimal.h"
#include "flang/Runtime/freestanding-tools.h"
#include <cinttypes>

namespace Fortran::runtime {
class Descriptor;
} // namespace Fortran::runtime

namespace Fortran::runtime::io {

class IoStatementState;

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
  RT_OFFLOAD_VAR_GROUP_BEGIN
  static constexpr char ListDirected{'g'}; // non-COMPLEX list-directed
  static constexpr char ListDirectedRealPart{'r'}; // emit "(r," or "(r;"
  static constexpr char ListDirectedImaginaryPart{'z'}; // emit "z)"
  static constexpr char ListDirectedNullValue{'n'}; // see 13.10.3.2
  static constexpr char DefinedDerivedType{'d'}; // DT defined I/O
  RT_OFFLOAD_VAR_GROUP_END
  constexpr RT_API_ATTRS bool IsListDirected() const {
    return descriptor == ListDirected || descriptor == ListDirectedRealPart ||
        descriptor == ListDirectedImaginaryPart;
  }
  constexpr RT_API_ATTRS bool IsNamelist() const {
    return IsListDirected() && modes.inNamelist;
  }

  char variation{'\0'}; // N, S, or X for EN, ES, EX; G/l for original G/list
  Fortran::common::optional<int> width; // the 'w' field; optional for A
  Fortran::common::optional<int> digits; // the 'm' or 'd' field
  Fortran::common::optional<int> expoDigits; // 'Ee' field
  MutableModes modes;
  int repeat{1};

  // "iotype" &/or "v_list" values for a DT'iotype'(v_list)
  // defined I/O data edit descriptor
  RT_OFFLOAD_VAR_GROUP_BEGIN
  static constexpr std::size_t maxIoTypeChars{32};
  static constexpr std::size_t maxVListEntries{4};
  RT_OFFLOAD_VAR_GROUP_END
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
  using CharType = char; // formats are always default kind CHARACTER

  RT_API_ATTRS FormatControl() {}
  RT_API_ATTRS FormatControl(const Terminator &, const CharType *format,
      std::size_t formatLength, const Descriptor *formatDescriptor = nullptr,
      int maxHeight = maxMaxHeight);

  // For attempting to allocate in a user-supplied stack area
  static RT_API_ATTRS std::size_t GetNeededSize(int maxHeight) {
    return sizeof(FormatControl) -
        sizeof(Iteration) * (maxMaxHeight - maxHeight);
  }

  // Extracts the next data edit descriptor, handling control edit descriptors
  // along the way.  If maxRepeat==0, this is a peek at the next data edit
  // descriptor.
  RT_API_ATTRS Fortran::common::optional<DataEdit> GetNextDataEdit(
      Context &, int maxRepeat = 1);

  // Emit any remaining character literals after the last data item (on output)
  // and perform remaining record positioning actions.
  RT_API_ATTRS void Finish(Context &);

private:
  RT_OFFLOAD_VAR_GROUP_BEGIN
  static constexpr std::uint8_t maxMaxHeight{100};

  struct Iteration {
    static constexpr int unlimited{-1};
    int start{0}; // offset in format_ of '(' or a repeated edit descriptor
    int remaining{0}; // while >0, decrement and iterate
  };
  RT_OFFLOAD_VAR_GROUP_END

  RT_API_ATTRS void SkipBlanks() {
    while (offset_ < formatLength_ &&
        (format_[offset_] == ' ' || format_[offset_] == '\t' ||
            format_[offset_] == '\v')) {
      ++offset_;
    }
  }
  RT_API_ATTRS CharType PeekNext() {
    SkipBlanks();
    return offset_ < formatLength_ ? format_[offset_] : '\0';
  }
  RT_API_ATTRS CharType GetNextChar(IoErrorHandler &handler) {
    SkipBlanks();
    if (offset_ >= formatLength_) {
      if (formatLength_ == 0) {
        handler.SignalError(
            IostatErrorInFormat, "Empty or badly assigned FORMAT");
      } else {
        handler.SignalError(
            IostatErrorInFormat, "FORMAT missing at least one ')'");
      }
      return '\n';
    }
    return format_[offset_++];
  }
  RT_API_ATTRS int GetIntField(
      IoErrorHandler &, CharType firstCh = '\0', bool *hadError = nullptr);

  // Advances through the FORMAT until the next data edit
  // descriptor has been found; handles control edit descriptors
  // along the way.  Returns the repeat count that appeared
  // before the descriptor (defaulting to 1) and leaves offset_
  // pointing to the data edit.
  RT_API_ATTRS int CueUpNextDataEdit(Context &, bool stop = false);

  static constexpr RT_API_ATTRS CharType Capitalize(CharType ch) {
    return ch >= 'a' && ch <= 'z' ? ch + 'A' - 'a' : ch;
  }

  RT_API_ATTRS void ReportBadFormat(
      Context &context, const char *msg, int offset) const {
    if constexpr (std::is_same_v<CharType, char>) {
      // Echo the bad format in the error message, but trim any leading or
      // trailing spaces.
      int firstNonBlank{0};
      while (firstNonBlank < formatLength_ && format_[firstNonBlank] == ' ') {
        ++firstNonBlank;
      }
      int lastNonBlank{formatLength_ - 1};
      while (lastNonBlank > firstNonBlank && format_[lastNonBlank] == ' ') {
        --lastNonBlank;
      }
      if (firstNonBlank <= lastNonBlank) {
        context.SignalError(IostatErrorInFormat,
            "%s; at offset %d in format '%.*s'", msg, offset,
            lastNonBlank - firstNonBlank + 1, format_ + firstNonBlank);
        return;
      }
    }
    context.SignalError(IostatErrorInFormat, "%s; at offset %d", msg, offset);
  }

  // Data members are arranged and typed so as to reduce size.
  // This structure may be allocated in stack space loaned by the
  // user program for internal I/O.
  const std::uint8_t maxHeight_{maxMaxHeight};
  std::uint8_t height_{0};
  bool freeFormat_{false};
  bool hitEnd_{false};
  const CharType *format_{nullptr};
  int formatLength_{0}; // in units of characters
  int offset_{0}; // next item is at format_[offset_]

  // must be last, may be incomplete
  Iteration stack_[maxMaxHeight];
};
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_FORMAT_H_
