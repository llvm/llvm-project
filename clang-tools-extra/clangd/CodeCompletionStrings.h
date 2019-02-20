//===--- CodeCompletionStrings.h ---------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for retrieving code completion information from
// `CodeCompletionString`.
//
//===----------------------------------------------------------------------===//

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

/// Similar to getDocComment, but returns the comment for a NamedDecl.
std::string getDeclComment(const ASTContext &Ctx, const NamedDecl &D);

/// Formats the signature for an item, as a display string and snippet.
/// e.g. for const_reference std::vector<T>::at(size_type) const, this returns:
///   *Signature = "(size_type) const"
///   *Snippet = "(${0:size_type})"
/// If set, RequiredQualifiers is the text that must be typed before the name.
/// e.g "Base::" when calling a base class member function that's hidden.
void getSignature(const CodeCompletionString &CCS, std::string *Signature,
                  std::string *Snippet,
                  std::string *RequiredQualifiers = nullptr);

/// Assembles formatted documentation for a completion result. This includes
/// documentation comments and other relevant information like annotations.
///
/// \param DocComment is a documentation comment for the original declaration,
///        it should be obtained via getDocComment or getParameterDocComment.
std::string formatDocumentation(const CodeCompletionString &CCS,
                                llvm::StringRef DocComment);

/// Gets detail to be used as the detail field in an LSP completion item. This
/// is usually the return type of a function.
std::string getReturnType(const CodeCompletionString &CCS);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_CODECOMPLETIONSTRINGS_H
