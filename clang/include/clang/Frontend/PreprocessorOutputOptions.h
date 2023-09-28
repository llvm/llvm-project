//===--- PreprocessorOutputOptions.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_PREPROCESSOROUTPUTOPTIONS_H
#define LLVM_CLANG_FRONTEND_PREPROCESSOROUTPUTOPTIONS_H

namespace clang {

/// PreprocessorOutputOptions - Options for controlling the C preprocessor
/// output (e.g., -E).
class PreprocessorOutputOptions {
public:
  unsigned ShowCPP : 1;            ///< Print normal preprocessed output.
  unsigned ShowComments : 1;       ///< Show comments.
  unsigned ShowLineMarkers : 1;    ///< Show \#line markers.
  unsigned UseLineDirectives : 1;   ///< Use \#line instead of GCC-style \# N.
  unsigned ShowMacroComments : 1;  ///< Show comments, even in macros.
  unsigned ShowMacros : 1;         ///< Print macro definitions.
  unsigned ShowIncludeDirectives : 1;  ///< Print includes, imports etc. within preprocessed output.
  unsigned ShowEmbedDirectives : 1;  ///< Print embeds, etc. within preprocessed output.
  unsigned RewriteIncludes : 1;    ///< Preprocess include directives only.
  unsigned RewriteImports  : 1;    ///< Include contents of transitively-imported modules.
  unsigned MinimizeWhitespace : 1; ///< Ignore whitespace from input.
  unsigned DirectivesOnly : 1; ///< Process directives but do not expand macros.
  unsigned KeepSystemIncludes : 1; ///< Do not expand system headers.

public:
  PreprocessorOutputOptions() {
    ShowCPP = 0;
    ShowComments = 0;
    ShowLineMarkers = 1;
    UseLineDirectives = 0;
    ShowMacroComments = 0;
    ShowMacros = 0;
    ShowIncludeDirectives = 0;
    ShowEmbedDirectives = 0;
    RewriteIncludes = 0;
    RewriteImports = 0;
    MinimizeWhitespace = 0;
    DirectivesOnly = 0;
    KeepSystemIncludes = 0;
  }
};

}  // end namespace clang

#endif
