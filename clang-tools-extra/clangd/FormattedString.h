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

#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace markup {

/// Holds text and knows how to lay it out. Multiple blocks can be grouped to
/// form a document. Blocks include their own trailing newlines, container
/// should trim them if need be.
class Block {
public:
  virtual void renderMarkdown(llvm::raw_ostream &OS) const = 0;
  virtual void renderPlainText(llvm::raw_ostream &OS) const = 0;
  std::string asMarkdown() const;
  std::string asPlainText() const;

  virtual ~Block() = default;
};

/// Represents parts of the markup that can contain strings, like inline code,
/// code block or plain text.
/// One must introduce different paragraphs to create separate blocks.
class Paragraph : public Block {
public:
  void renderMarkdown(llvm::raw_ostream &OS) const override;
  void renderPlainText(llvm::raw_ostream &OS) const override;

  /// Append plain text to the end of the string.
  Paragraph &appendText(std::string Text);

  /// Append inline code, this translates to the ` block in markdown.
  Paragraph &appendCode(std::string Code);

private:
  struct Chunk {
    enum {
      PlainText,
      InlineCode,
    } Kind = PlainText;
    std::string Contents;
    /// Language for code block chunks. Ignored for other chunks.
    std::string Language;
  };
  std::vector<Chunk> Chunks;
};

/// A format-agnostic representation for structured text. Allows rendering into
/// markdown and plaintext.
class Document {
public:
  /// Adds a semantical block that will be separate from others.
  Paragraph &addParagraph();
  /// Inserts a vertical space into the document.
  void addSpacer();
  /// Adds a block of code. This translates to a ``` block in markdown. In plain
  /// text representation, the code block will be surrounded by newlines.
  void addCodeBlock(std::string Code, std::string Language = "cpp");

  std::string asMarkdown() const;
  std::string asPlainText() const;

private:
  std::vector<std::unique_ptr<Block>> Children;
};

} // namespace markup
} // namespace clangd
} // namespace clang

#endif
