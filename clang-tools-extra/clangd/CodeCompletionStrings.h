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
class ASTContext;

namespace clangd {

/// Gets a minimally formatted documentation comment of \p Result, with comment
/// markers stripped. See clang::RawComment::getFormattedText() for the detailed
/// explanation of how the comment text is transformed.
/// Returns empty string when no comment is available.
std::string getDocComment(const ASTContext &Ctx,
                          const CodeCompletionResult &Result);

/// Gets a minimally formatted documentation for parameter of \p Result,
/// corresponding to argument number \p ArgIndex.
/// This currently looks for comments attached to the parameter itself, and
/// doesn't extract them from function documentation.
/// Returns empty string when no comment is available.
std::string
getParameterDocComment(const ASTContext &Ctx,
                       const CodeCompleteConsumer::OverloadCandidate &Result,
                       unsigned ArgIndex);

/// Gets label and insert text for a completion item. For example, for function
/// `Foo`, this returns <"Foo(int x, int y)", "Foo"> without snippts enabled.
///
/// If \p EnableSnippets is true, this will try to use snippet for the insert
/// text. Otherwise, the insert text will always be plain text.
void getLabelAndInsertText(const CodeCompletionString &CCS, std::string *Label,
                           std::string *InsertText, bool EnableSnippets);

/// Assembles formatted documentation for a completion result. This includes
/// documentation comments and other relevant information like annotations.
///
/// \param DocComment is a documentation comment for the original declaration,
///        it should be obtained via getDocComment or getParameterDocComment.
std::string formatDocumentation(const CodeCompletionString &CCS,
                                llvm::StringRef DocComment);

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
