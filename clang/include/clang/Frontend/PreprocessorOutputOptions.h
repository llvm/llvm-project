//===--- PreprocessorOutputOptions.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  unsigned ShowMacroComments : 1;  ///< Show comments, even in macros.
  unsigned ShowMacros : 1;         ///< Print macro definitions.
  unsigned RewriteIncludes : 1;    ///< Preprocess include directives only.

public:
  PreprocessorOutputOptions() {
    ShowCPP = 1;
    ShowComments = 0;
    ShowLineMarkers = 1;
    ShowMacroComments = 0;
    ShowMacros = 0;
    RewriteIncludes = 0;
  }
};

}  // end namespace clang

#endif
