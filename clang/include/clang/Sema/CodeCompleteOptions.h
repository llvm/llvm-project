//===---- CodeCompleteOptions.h - Code Completion Options -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_CODECOMPLETEOPTIONS_H
#define LLVM_CLANG_SEMA_CODECOMPLETEOPTIONS_H

namespace clang {

/// Options controlling the behavior of code completion.
class CodeCompleteOptions {
public:
#define CODE_COMPLETEOPT(Name, Bits, Description) unsigned Name : Bits;
#define TYPED_CODE_COMPLETEOPT(Type, Name, Description) Type Name;
#include "clang/Sema/CodeCompleteOptions.def"

  CodeCompleteOptions()
      : IncludeMacros(0), IncludeCodePatterns(0), IncludeGlobals(1),
        IncludeNamespaceLevelDecls(1), IncludeBriefComments(0), LoadExternal(1),
        IncludeFixIts(0) {}
};

} // namespace clang

#endif

