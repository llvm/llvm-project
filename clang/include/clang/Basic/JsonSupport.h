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
#include "llvm/Support/raw_ostream.h"


namespace clang {

inline raw_ostream &Indent(raw_ostream &Out, const unsigned int Space,
                           bool IsDot) {
  for (unsigned int I = 0; I < Space * 2; ++I)
    Out << (IsDot ? "&nbsp;" : " ");
  return Out;
}

} // namespace clang

#endif // LLVM_CLANG_BASIC_JSONSUPPORT_H
