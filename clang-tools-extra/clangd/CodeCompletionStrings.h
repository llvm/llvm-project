//===--- CodeCompletionStrings.h ---------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Functions for retrieving code completion information from
// `CodeCompletionString`.
//
//===---------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETIONSTRINGS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETIONSTRINGS_H

#include "clang/Sema/CodeCompleteConsumer.h"

namespace clang {
namespace clangd {

/// Gets label and insert text for a completion item. For example, for function
/// `Foo`, this returns <"Foo(int x, int y)", "Foo"> without snippts enabled.
///
/// If \p EnableSnippets is true, this will try to use snippet for the insert
/// text. Otherwise, the insert text will always be plain text.
void getLabelAndInsertText(const CodeCompletionString &CCS, std::string *Label,
                           std::string *InsertText, bool EnableSnippets);

/// Gets the documentation for a completion item. For example, comment for the
/// a class declaration.
std::string getDocumentation(const CodeCompletionString &CCS);

/// Gets detail to be used as the detail field in an LSP completion item. This
/// is usually the return type of a function.
std::string getDetail(const CodeCompletionString &CCS);

/// Gets the piece of text that the user is expected to type to match the
/// code-completion string, typically a keyword or the name of a declarator or
/// macro.
std::string getFilterText(const CodeCompletionString &CCS);

} // namespace clangd
} // namespace clang

#endif
