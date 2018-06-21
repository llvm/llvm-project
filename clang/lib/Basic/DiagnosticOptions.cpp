//===--- DiagnosticOptions.cpp - C Language Family Diagnostic Handling ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the DiagnosticOptions related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

raw_ostream &operator<<(raw_ostream &Out, DiagnosticLevelMask M) {
  using UT = std::underlying_type<DiagnosticLevelMask>::type;
  return Out << static_cast<UT>(M);
}

} // end namespace clang
