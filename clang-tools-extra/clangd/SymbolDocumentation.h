//===--- SymbolDocumentation.h ==---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class to parse doxygen comments into a flat structure for consumption
// in e.g. Hover and Code Completion
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentVisitor.h"

namespace clang {
namespace clangd {

template <class String> struct ParameterDocumentation {
  String Name;
  String Description;

  ParameterDocumentation<llvm::StringRef> toRef() const;
  ParameterDocumentation<std::string> toOwned() const;
};

using ParameterDocumentationRef = ParameterDocumentation<llvm::StringRef>;
using ParameterDocumentationOwned = ParameterDocumentation<std::string>;

/// @brief Represents a parsed doxygen comment.
/// @details Currently there's special handling for the "brief", "param"
/// "returns", "note" and "warning" commands. The content of all other
/// paragraphs will be appended to the #Description field.
/// If you're only interested in the full comment, but with comment
/// markers stripped, use the #CommentText field.
/// \tparam String When built from a declaration, we're building the strings
/// by ourselves, so in this case String==std::string.
/// However, when storing the contents of this class in the index, we need to
/// use llvm::StringRef. To connvert between std::string and llvm::StringRef
/// versions of this class, use toRef() and toOwned().
template <class String> class SymbolDocumentation {
public:
  friend class CommentToSymbolDocumentation;

  static SymbolDocumentation<String> descriptionOnly(String &&Description) {
    SymbolDocumentation<String> Doc;
    Doc.Description = Description;
    Doc.CommentText = Description;
    return Doc;
  }

  /// Constructs with all fields as empty strings/vectors.
  SymbolDocumentation() = default;

  SymbolDocumentation<llvm::StringRef> toRef() const;
  SymbolDocumentation<std::string> toOwned() const;

  bool empty() const { return CommentText.empty(); }

  /// Paragraph of the "brief" command.
  String Brief;

  /// Paragraph of the "return" command.
  String Returns;

  /// Paragraph(s) of the "note" command(s)
  llvm::SmallVector<String, 1> Notes;
  /// Paragraph(s) of the "warning" command(s)
  llvm::SmallVector<String, 1> Warnings;

  /// Parsed paragaph(s) of the "param" comamnd(s)
  llvm::SmallVector<ParameterDocumentation<String>> Parameters;

  /// All the paragraphs we don't have any special handling for,
  /// e.g. "details".
  String Description;

  /// The full documentation comment with comment markers stripped.
  /// See clang::RawComment::getFormattedText() for the detailed
  /// explanation of how the comment text is transformed.
  String CommentText;
};

using SymbolDocumentationOwned = SymbolDocumentation<std::string>;
using SymbolDocumentationRef = SymbolDocumentation<llvm::StringRef>;

/// @param RC the comment to parse
/// @param D the declaration that \p RC belongs to
/// @return parsed doxgen documentation.
SymbolDocumentationOwned
parseDoxygenComment(const RawComment &RC, const ASTContext &Ctx, const Decl *D);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_SYMBOLDOCUMENTATION_H
