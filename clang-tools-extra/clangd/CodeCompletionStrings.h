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
/// If \p CommentsFromHeaders parameter is set, only comments from the main
/// file will be returned. It is used to workaround crashes when parsing
/// comments in the stale headers, coming from completion preamble.
std::string getDocComment(const ASTContext &Ctx,
                          const CodeCompletionResult &Result,
                          bool CommentsFromHeaders);

/// Gets a minimally formatted documentation for parameter of \p Result,
/// corresponding to argument number \p ArgIndex.
/// This currently looks for comments attached to the parameter itself, and
/// doesn't extract them from function documentation.
/// Returns empty string when no comment is available.
/// If \p CommentsFromHeaders parameter is set, only comments from the main
/// file will be returned. It is used to workaround crashes when parsing
/// comments in the stale headers, coming from completion preamble.
std::string
getParameterDocComment(const ASTContext &Ctx,
                       const CodeCompleteConsumer::OverloadCandidate &Result,
                       unsigned ArgIndex, bool CommentsFromHeaders);

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

} // namespace clangd
} // namespace clang

#endif
