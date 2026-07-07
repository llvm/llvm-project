//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares the Markdown parser entry point and the ASTContext that owns the
/// lifetime of the nodes it produces.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_MARKDOWNPARSER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_MARKDOWNPARSER_H

#include "Markdown.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <type_traits>
#include <utility>

namespace clang::doc::markdown {

/// Restricts ASTContext::allocate to Markdown node types.
template <typename T>
using IsMarkdownNode = std::enable_if_t<std::is_base_of_v<InlineNode, T> ||
                                        std::is_base_of_v<BlockNode, T> ||
                                        std::is_same_v<T, ListItemNode>>;

/// Owns the bump-pointer arena backing every node produced by the parser.
/// Nodes allocated through allocate() live as long as the ASTContext.
class ASTContext {
public:
  ASTContext() = default;

  template <typename T, typename... Args, typename = IsMarkdownNode<T>>
  T *allocate(Args &&...Params) {
    return new (Arena.Allocate<T>()) T(std::forward<Args>(Params)...);
  }

  /// Copies a string into the arena so its lifetime matches the ASTContext.
  /// Used when the parser needs to synthesize text that is not a contiguous
  /// slice of the original buffer (e.g. joined paragraph lines).
  llvm::StringRef intern(llvm::StringRef S) {
    char *Buf = Arena.Allocate<char>(S.size());
    llvm::copy(S, Buf);
    return llvm::StringRef(Buf, S.size());
  }

private:
  llvm::BumpPtrAllocator Arena;
};

/// Parses Markdown text into a DocumentNode owned by Ctx.
DocumentNode *parseMarkdown(llvm::StringRef Text, ASTContext &Ctx);

} // namespace clang::doc::markdown

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_MARKDOWN_MARKDOWNPARSER_H
