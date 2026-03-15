//===--- NumericLiteralCaseFixer.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares NumericLiteralCaseFixer that standardizes character case
/// within numeric literals.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_NUMERICLITERALCASEFIXER_H
#define LLVM_CLANG_LIB_FORMAT_NUMERICLITERALCASEFIXER_H

#include "TokenAnalyzer.h"

namespace clang {
namespace format {

class NumericLiteralCaseFixer {
public:
  std::pair<tooling::Replacements, unsigned> process(const Environment &Env,
                                                     const FormatStyle &Style);
};

} // end namespace format
} // end namespace clang

#endif
