//===- clang/Lex/DependencyDirectivesScanner.h ---------------------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is the interface for scanning header and source files to get the
/// minimum necessary preprocessor directives for evaluating includes. It
/// reduces the source down to #define, #include, #import, @import, and any
/// conditional preprocessor logic that contains one of those.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_DEPENDENCYDIRECTIVESSCANNER_H
#define LLVM_CLANG_LEX_DEPENDENCYDIRECTIVESSCANNER_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class DiagnosticsEngine;

namespace dependency_directives_scan {

/// Represents the kind of preprocessor directive or a module declaration that
/// is tracked by the scanner in its token output.
enum DirectiveKind : uint8_t {
  pp_none,
  pp_include,
  pp___include_macros,
  pp_define,
  pp_undef,
  pp_import,
  pp_pragma_import,
  pp_pragma_once,
  pp_pragma_push_macro,
  pp_pragma_pop_macro,
  pp_pragma_include_alias,
  pp_include_next,
  pp_if,
  pp_ifdef,
  pp_ifndef,
  pp_elif,
  pp_elifdef,
  pp_elifndef,
  pp_else,
  pp_endif,
  decl_at_import,
  cxx_export_decl,
  cxx_module_decl,
  cxx_import_decl,
  pp_eof,
};

/// Represents a directive that's lexed as part of the dependency directives
/// scanning. It's used to track various preprocessor directives that could
/// potentially have an effect on the depedencies.
struct Directive {
  /// The kind of token.
  DirectiveKind Kind = pp_none;

  /// Offset into the output byte stream of where the directive begins.
  int Offset = -1;

  Directive(DirectiveKind K, int Offset) : Kind(K), Offset(Offset) {}
};

/// Simplified token range to track the range of a potentially skippable PP
/// directive.
struct SkippedRange {
  /// Offset into the output byte stream of where the skipped directive begins.
  int Offset;

  /// The number of bytes that can be skipped before the preprocessing must
  /// resume.
  int Length;
};

/// Computes the potential source ranges that can be skipped by the preprocessor
/// when skipping a directive like #if, #ifdef or #elsif.
///
/// \returns false on success, true on error.
bool computeSkippedRanges(ArrayRef<Directive> Input,
                          llvm::SmallVectorImpl<SkippedRange> &Range);

} // end namespace dependency_directives_scan

/// Minimize the input down to the preprocessor directives that might have
/// an effect on the dependencies for a compilation unit.
///
/// This function deletes all non-preprocessor code, and strips anything that
/// can't affect what gets included. It canonicalizes whitespace where
/// convenient to stabilize the output against formatting changes in the input.
///
/// Clears the output vectors at the beginning of the call.
///
/// \returns false on success, true on error. If the diagnostic engine is not
/// null, an appropriate error is reported using the given input location
/// with the offset that corresponds to the minimizer's current buffer offset.
bool scanSourceForDependencyDirectives(
    llvm::StringRef Input, llvm::SmallVectorImpl<char> &Output,
    llvm::SmallVectorImpl<dependency_directives_scan::Directive> &Directives,
    DiagnosticsEngine *Diags = nullptr,
    SourceLocation InputSourceLoc = SourceLocation());

} // end namespace clang

#endif // LLVM_CLANG_LEX_DEPENDENCYDIRECTIVESSCANNER_H
