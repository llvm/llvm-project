//===--- MacroArgs.h - Formal argument info for Macros ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroArgs interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PPDIRECTIVEPARAMETER_H
#define LLVM_CLANG_LEX_PPDIRECTIVEPARAMETER_H

#include "clang/Basic/SourceLocation.h"

namespace clang {

/// Captures basic information about a preprocessor directive parameter.
class PPDirectiveParameter {
public:
  SourceLocation Start;
  SourceLocation End;

  PPDirectiveParameter(SourceLocation Start, SourceLocation End)
      : Start(Start), End(End) {}
};

} // end namespace clang

#endif
