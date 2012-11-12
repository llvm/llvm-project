//===---- CodeCompleteOptions.h - Code Completion Options -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_CODECOMPLETEOPTIONS_H
#define LLVM_CLANG_SEMA_CODECOMPLETEOPTIONS_H

/// Options controlling the behavior of code completion.
class CodeCompleteOptions {
public:
  ///< Show macros in code completion results.
  unsigned IncludeMacros : 1;

  ///< Show code patterns in code completion results.
  unsigned IncludeCodePatterns : 1;

  ///< Show top-level decls in code completion results.
  unsigned IncludeGlobals : 1;

  ///< Show brief documentation comments in code completion results.
  unsigned IncludeBriefComments : 1;

  CodeCompleteOptions() :
      IncludeMacros(0),
      IncludeCodePatterns(0),
      IncludeGlobals(1),
      IncludeBriefComments(0)
  { }
};

#endif

