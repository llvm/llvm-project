//===--- Rewriters.h - Rewriter implementations     -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This header contains miscellaneous utilities for various front-end actions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REWRITE_FRONTEND_REWRITERS_H
#define LLVM_CLANG_REWRITE_FRONTEND_REWRITERS_H

#include "clang/Basic/LLVM.h"
#include "clang/Support/Compiler.h"

namespace clang {
class Preprocessor;
class PreprocessorOutputOptions;

/// RewriteMacrosInInput - Implement -rewrite-macros mode.
CLANG_ABI void RewriteMacrosInInput(Preprocessor &PP, raw_ostream *OS);

/// DoRewriteTest - A simple test for the TokenRewriter class.
CLANG_ABI void DoRewriteTest(Preprocessor &PP, raw_ostream *OS);

/// RewriteIncludesInInput - Implement -frewrite-includes mode.
CLANG_ABI void RewriteIncludesInInput(Preprocessor &PP, raw_ostream *OS,
                            const PreprocessorOutputOptions &Opts);

}  // end namespace clang

#endif
