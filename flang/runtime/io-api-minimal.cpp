//===-- runtime/io-api-minimal.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements the subset of the I/O statement API needed for basic
// list-directed output (PRINT *) of intrinsic types.

#include "edit-output.h"
#include "format.h"
#include "io-api-common.h"
#include "io-stmt.h"
#include "terminator.h"
#include "tools.h"
#include "unit.h"
#include "flang/Runtime/io-api.h"

namespace Fortran::runtime::io {
RT_EXT_API_GROUP_BEGIN

Cookie IODEF(BeginExternalListOutput)(
    ExternalUnit unitNumber, const char *sourceFile, int sourceLine) {
  return BeginExternalListIO<Direction::Output, ExternalListIoStatementState>(
      unitNumber, sourceFile, sourceLine);
}

enum Iostat IODEF(EndIoStatement)(Cookie cookie) {
  IoStatementState &io{*cookie};
  return static_cast<enum Iostat>(io.EndIoStatement());
}

template <int KIND, typename INT = CppTypeFor<TypeCategory::Integer, KIND>>
inline RT_API_ATTRS bool FormattedScalarIntegerOutput(
    IoStatementState &io, INT x, const char *whence) {
  if (io.CheckFormattedStmtType<Direction::Output>(whence)) {
    auto edit{io.GetNextDataEdit()};
    return edit && EditIntegerOutput<KIND>(io, *edit, x, /*isSigned=*/true);
  } else {
    return false;
  }
}

bool IODEF(OutputInteger8)(Cookie cookie, std::int8_t n) {
  return FormattedScalarIntegerOutput<1>(*cookie, n, "OutputInteger8");
}

bool IODEF(OutputInteger16)(Cookie cookie, std::int16_t n) {
  return FormattedScalarIntegerOutput<2>(*cookie, n, "OutputInteger16");
}

bool IODEF(OutputInteger32)(Cookie cookie, std::int32_t n) {
  return FormattedScalarIntegerOutput<4>(*cookie, n, "OutputInteger32");
}

bool IODEF(OutputInteger64)(Cookie cookie, std::int64_t n) {
  return FormattedScalarIntegerOutput<8>(*cookie, n, "OutputInteger64");
}

#ifdef __SIZEOF_INT128__
bool IODEF(OutputInteger128)(Cookie cookie, common::int128_t n) {
  return FormattedScalarIntegerOutput<16>(*cookie, n, "OutputInteger128");
}
#endif

template <int KIND,
    typename REAL = typename RealOutputEditing<KIND>::BinaryFloatingPoint>
inline RT_API_ATTRS bool FormattedScalarRealOutput(
    IoStatementState &io, REAL x, const char *whence) {
  if (io.CheckFormattedStmtType<Direction::Output>(whence)) {
    auto edit{io.GetNextDataEdit()};
    return edit && RealOutputEditing<KIND>{io, x}.Edit(*edit);
  } else {
    return false;
  }
}

bool IODEF(OutputReal32)(Cookie cookie, float x) {
  return FormattedScalarRealOutput<4>(*cookie, x, "OutputReal32");
}

bool IODEF(OutputReal64)(Cookie cookie, double x) {
  return FormattedScalarRealOutput<8>(*cookie, x, "OutputReal64");
}

template <int KIND,
    typename REAL = typename RealOutputEditing<KIND>::BinaryFloatingPoint>
inline RT_API_ATTRS bool FormattedScalarComplexOutput(
    IoStatementState &io, REAL re, REAL im, const char *whence) {
  if (io.CheckFormattedStmtType<Direction::Output>(whence)) {
    if (io.get_if<ListDirectedStatementState<Direction::Output>>() != nullptr) {
      DataEdit rEdit, iEdit;
      rEdit.descriptor = DataEdit::ListDirectedRealPart;
      iEdit.descriptor = DataEdit::ListDirectedImaginaryPart;
      rEdit.modes = iEdit.modes = io.mutableModes();
      return RealOutputEditing<KIND>{io, re}.Edit(rEdit) &&
          RealOutputEditing<KIND>{io, im}.Edit(iEdit);
    } else {
      auto reEdit{io.GetNextDataEdit()};
      if (reEdit && RealOutputEditing<KIND>{io, re}.Edit(*reEdit)) {
        auto imEdit{io.GetNextDataEdit()};
        return imEdit && RealOutputEditing<KIND>{io, im}.Edit(*imEdit);
      }
    }
  }
  return false;
}

bool IODEF(OutputComplex32)(Cookie cookie, float re, float im) {
  return FormattedScalarComplexOutput<4>(*cookie, re, im, "OutputComplex32");
}

bool IODEF(OutputComplex64)(Cookie cookie, double re, double im) {
  return FormattedScalarComplexOutput<8>(*cookie, re, im, "OutputComplex64");
}

bool IODEF(OutputAscii)(Cookie cookie, const char *x, std::size_t length) {
  IoStatementState &io{*cookie};
  if (!x) {
    io.GetIoErrorHandler().Crash("Null address for character output item");
  } else if (auto *listOutput{
                 io.get_if<ListDirectedStatementState<Direction::Output>>()}) {
    return ListDirectedCharacterOutput(io, *listOutput, x, length);
  } else if (io.CheckFormattedStmtType<Direction::Output>("OutputAscii")) {
    auto edit{io.GetNextDataEdit()};
    return edit && EditCharacterOutput(io, *edit, x, length);
  } else {
    return false;
  }
}

bool IODEF(OutputLogical)(Cookie cookie, bool truth) {
  IoStatementState &io{*cookie};
  if (auto *listOutput{
          io.get_if<ListDirectedStatementState<Direction::Output>>()}) {
    return ListDirectedLogicalOutput(io, *listOutput, truth);
  } else if (io.CheckFormattedStmtType<Direction::Output>("OutputAscii")) {
    auto edit{io.GetNextDataEdit()};
    return edit && EditLogicalOutput(io, *edit, truth);
  } else {
    return false;
  }
}

} // namespace Fortran::runtime::io

#if defined(_LIBCPP_VERBOSE_ABORT)
// Provide own definition for `std::__libcpp_verbose_abort` to avoid dependency
// on the version provided by libc++.

void std::__libcpp_verbose_abort(char const *format, ...) {
  va_list list;
  va_start(list, format);
  std::vfprintf(stderr, format, list);
  va_end(list);

  std::abort();
}
#endif

RT_EXT_API_GROUP_END
