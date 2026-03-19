//===--- Markup.h -------------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A model of formatted text that can be rendered to plaintext or markdown.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_MARKUP_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SUPPORT_MARKUP_H

#include "llvm/Support/raw_ostream.h"
#include <cstddef>
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
  virtual void renderEscapedMarkdown(llvm::raw_ostream &OS) const = 0;
  virtual void renderMarkdown(llvm::raw_ostream &OS) const = 0;
  virtual void renderPlainText(llvm::raw_ostream &OS) const = 0;
  virtual std::unique_ptr<Block> clone() const = 0;
  std::string asEscapedMarkdown() const;
  std::string asMarkdown() const;
  std::string asPlainText() const;

  virtual bool isRuler() const { return false; }
  virtual ~Block() = default;
};

/// Represents parts of the markup that can contain strings, like inline code,
/// code block or plain text.
/// One must introduce different paragraphs to create separate blocks.
class Paragraph : public Block {
public:
  void renderEscapedMarkdown(llvm::raw_ostream &OS) const override;
  void renderMarkdown(llvm::raw_ostream &OS) const override;
  void renderPlainText(llvm::raw_ostream &OS) const override;
  std::unique_ptr<Block> clone() const override;

  /// Append plain text to the end of the string.
  Paragraph &appendText(llvm::StringRef Text);

  /// Append emphasized text, this translates to the * block in markdown.
  Paragraph &appendEmphasizedText(llvm::StringRef Text);

  /// Append bold text, this translates to the ** block in markdown.
  Paragraph &appendBoldText(llvm::StringRef Text);

  /// Append inline code, this translates to the ` block in markdown.
  /// \p Preserve indicates the code span must be apparent even in plaintext.
  Paragraph &appendCode(llvm::StringRef Code, bool Preserve = false);

  /// Ensure there is space between the surrounding chunks.
  /// Has no effect at the beginning or end of a paragraph.
  Paragraph &appendSpace();

private:
  enum ChunkKind { PlainText, InlineCode, Bold, Emphasized };
  struct Chunk {
    ChunkKind Kind = PlainText;
    // Preserve chunk markers in plaintext.
    bool Preserve = false;
    std::string Contents;
    // Whether this chunk should be surrounded by whitespace.
    // Consecutive SpaceAfter and SpaceBefore will be collapsed into one space.
    // Code spans don't usually set this: their spaces belong "inside" the span.
    bool SpaceBefore = false;
    bool SpaceAfter = false;
  };
  std::vector<Chunk> Chunks;

  /// Estimated size of the string representation of this paragraph.
  /// Used to reserve space in the output string.
  /// Each time paragraph content is added, this value is updated.
  /// This is an estimate, so it may not be accurate but can help
  /// reducing dynamically reallocating string memory.
  unsigned EstimatedStringSize = 0;

  Paragraph &appendChunk(llvm::StringRef Contents, ChunkKind K);

  llvm::StringRef chooseMarker(llvm::ArrayRef<llvm::StringRef> Options,
                               llvm::StringRef Text) const;

  /// Checks if the given line ends with punctuation that indicates a line break
  /// (.:,;!?).
  ///
  /// If \p IsMarkdown is false, lines ending with 2 spaces are also considered
  /// as indicating a line break. This is not needed for markdown because the
  /// client renderer will handle this case.
  bool punctuationIndicatesLineBreak(llvm::StringRef Line,
                                     bool IsMarkdown) const;

  /// Checks if the given line starts with a hard line break indicator.
  ///
  /// If \p IsMarkdown is true, only '@' and '\' are considered as indicators.
  /// Otherwise, '-', '*', '@', '\', '>', '#', '`' and a digit followed by '.'
  /// or ')' are also considered as indicators.
  bool isHardLineBreakIndicator(llvm::StringRef Rest, bool IsMarkdown) const;

  /// Checks if a hard line break should be added after the given line.
  bool isHardLineBreakAfter(llvm::StringRef Line, llvm::StringRef Rest,
                            bool IsMarkdown) const;

  /// \brief Go through the contents line by line to handle the newlines
  /// and required spacing correctly for markdown rendering.
  ///
  /// Newlines are added if:
  /// - the line ends with punctuation that indicates a line break (.:,;!?)
  /// - the next line starts with a hard line break indicator \\ (escaped
  /// markdown/doxygen command) or @ (doxygen command)
  ///
  /// This newline handling is only used when the client requests markdown
  /// for hover/signature help content.
  /// Markdown does not add any newlines inside paragraphs unless the user
  /// explicitly adds them. For hover/signature help content, we still want to
  /// add newlines in some cases to improve readability, especially when doxygen
  /// parsing is disabled or not implemented (like for signature help).
  /// Therefore we add newlines in the above mentioned cases.
  ///
  /// In addition to that, we need to consider that the user can configure
  /// clangd to treat documentation comments as plain text, while the client
  /// requests markdown.
  /// In this case, all markdown syntax is escaped and will
  /// not be rendered as expected by markdown.
  /// Examples are lists starting with '-' or headings starting with '#'.
  /// With the above next line heuristics, these cases are also covered by the
  /// '\\' new line indicator.
  ///
  /// FIXME: The heuristic fails e.g. for lists starting with '*' because it is
  /// also used for emphasis in markdown and should not be treated as a newline.
  ///
  /// \param OS The stream to render to.
  /// \param ParagraphText The text of the paragraph to render.
  void renderNewlinesMarkdown(llvm::raw_ostream &OS,
                              llvm::StringRef ParagraphText) const;

  /// \brief Go through the contents line by line to handle the newlines
  /// and required spacing correctly for plain text rendering.
  ///
  /// Newlines are added if:
  /// - the line ends with 2 spaces and a newline follows
  /// - the line ends with punctuation that indicates a line break (.:,;!?)
  /// - the next line starts with a hard line break indicator (-@>#`\\ or a
  ///   digit followed by '.' or ')'), ignoring leading whitespace.
  ///
  /// Otherwise, newlines in the input are replaced with a single space.
  ///
  /// Multiple spaces are collapsed into a single space.
  ///
  /// Lines containing only whitespace are ignored.
  ///
  /// This newline handling is only used when the client requests plain
  /// text for hover/signature help content.
  /// Therefore with this approach we mimic the behavior of markdown rendering
  /// for these clients.
  ///
  /// \param OS The stream to render to.
  /// \param ParagraphText The text of the paragraph to render.
  void renderNewlinesPlaintext(llvm::raw_ostream &OS,
                               llvm::StringRef ParagraphText) const;
};

/// Represents a sequence of one or more documents. Knows how to print them in a
/// list like format, e.g. by prepending with "- " and indentation.
class BulletList : public Block {
public:
  BulletList();
  ~BulletList();

  // A BulletList rendered in markdown is a tight list if it is not a nested
  // list and no item contains multiple paragraphs. Otherwise, it is a loose
  // list.
  void renderEscapedMarkdown(llvm::raw_ostream &OS) const override;
  void renderMarkdown(llvm::raw_ostream &OS) const override;
  void renderPlainText(llvm::raw_ostream &OS) const override;
  std::unique_ptr<Block> clone() const override;

  class Document &addItem();

private:
  std::vector<class Document> Items;
};

/// A format-agnostic representation for structured text. Allows rendering into
/// markdown and plaintext.
class Document {
public:
  Document() = default;
  Document(const Document &Other) { *this = Other; }
  Document &operator=(const Document &);
  Document(Document &&) = default;
  Document &operator=(Document &&) = default;

  void append(Document Other);

  /// Adds a semantical block that will be separate from others.
  Paragraph &addParagraph();
  /// Inserts a horizontal separator to the document.
  void addRuler();
  /// Adds a block of code. This translates to a ``` block in markdown. In plain
  /// text representation, the code block will be surrounded by newlines.
  void addCodeBlock(std::string Code, std::string Language = "cpp");
  /// Heading is a special type of paragraph that will be prepended with \p
  /// Level many '#'s in markdown.
  Paragraph &addHeading(size_t Level);

  BulletList &addBulletList();

  /// Doesn't contain any trailing newlines and escaped markdown syntax.
  /// It is expected that the result of this function
  /// is rendered as markdown.
  std::string asEscapedMarkdown() const;
  /// Doesn't contain any trailing newlines.
  /// It is expected that the result of this function
  /// is rendered as markdown.
  std::string asMarkdown() const;
  /// Doesn't contain any trailing newlines.
  std::string asPlainText() const;

private:
  std::vector<std::unique_ptr<Block>> Children;
};
} // namespace markup
} // namespace clangd
} // namespace clang

#endif
