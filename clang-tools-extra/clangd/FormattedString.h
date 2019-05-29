//===--- FormattedString.h ----------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple intermediate representation of formatted text that could be
// converted to plaintext or markdown.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FORMATTEDSTRING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FORMATTEDSTRING_H

#include <string>
#include <vector>

namespace clang {
namespace clangd {

/// A structured string representation that could be converted to markdown or
/// plaintext upon requrest.
class FormattedString {
public:
  /// Append plain text to the end of the string.
  void appendText(std::string Text);
  /// Append a block of C++ code. This translates to a ``` block in markdown.
  /// In a plain text representation, the code block will be surrounded by
  /// newlines.
  void appendCodeBlock(std::string Code, std::string Language = "cpp");
  /// Append an inline block of C++ code. This translates to the ` block in
  /// markdown.
  void appendInlineCode(std::string Code);

  std::string renderAsMarkdown() const;
  std::string renderAsPlainText() const;
  std::string renderForTests() const;

private:
  enum class ChunkKind {
    PlainText,       /// A plain text paragraph.
    CodeBlock,       /// A block of code.
    InlineCodeBlock, /// An inline block of code.
  };
  struct Chunk {
    ChunkKind Kind = ChunkKind::PlainText;
    std::string Contents;
    /// Language for code block chunks. Ignored for other chunks.
    std::string Language;
  };
  std::vector<Chunk> Chunks;
};

} // namespace clangd
} // namespace clang

#endif
