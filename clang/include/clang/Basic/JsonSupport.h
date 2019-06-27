//===- JsonSupport.h - JSON Output Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_JSONSUPPORT_H
#define LLVM_CLANG_BASIC_JSONSUPPORT_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"


namespace clang {

inline raw_ostream &Indent(raw_ostream &Out, const unsigned int Space,
                           bool IsDot) {
  for (unsigned int I = 0; I < Space * 2; ++I)
    Out << (IsDot ? "&nbsp;" : " ");
  return Out;
}

inline std::string JsonFormat(StringRef RawSR, bool AddQuotes) {
  if (RawSR.empty())
    return "null";

  // Trim special characters.
  std::string Str = RawSR.trim().str();
  size_t Pos = 0;

  // Escape double quotes.
  while (true) {
    Pos = Str.find('\"', Pos);
    if (Pos == std::string::npos)
      break;

    // Prevent bad conversions.
    size_t TempPos = (Pos != 0) ? Pos - 1 : 0;

    // See whether the current double quote is escaped.
    if (TempPos != Str.find("\\\"", TempPos)) {
      Str.insert(Pos, "\\");
      ++Pos; // As we insert the escape-character move plus one.
    }

    ++Pos;
  }

  // Remove new-lines.
  Str.erase(std::remove(Str.begin(), Str.end(), '\n'), Str.end());

  if (!AddQuotes)
    return Str;

  return '\"' + Str + '\"';
}

} // namespace clang

#endif // LLVM_CLANG_BASIC_JSONSUPPORT_H
