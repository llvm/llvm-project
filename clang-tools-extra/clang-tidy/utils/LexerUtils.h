//===--- LexerUtils.h - clang-tidy-------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H

#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace tidy {
namespace lexer_utils {

// Returns previous non-comment token skipping over any comment text or
// tok::unknown if not found.
Token getPreviousNonCommentToken(const ASTContext &Context,
                                 SourceLocation Location);

} // namespace lexer_utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_LEXER_UTILS_H
