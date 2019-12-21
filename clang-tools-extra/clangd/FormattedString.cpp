//===--- FormattedString.cpp --------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FormattedString.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace markup {

namespace {
/// Escape a markdown text block. Ensures the punctuation will not introduce
/// any of the markdown constructs.
std::string renderText(llvm::StringRef Input) {
  // Escaping ASCII punctuation ensures we can't start a markdown construct.
  constexpr llvm::StringLiteral Punctuation =
      R"txt(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)txt";

  std::string R;
  for (size_t From = 0; From < Input.size();) {
    size_t Next = Input.find_first_of(Punctuation, From);
    R += Input.substr(From, Next - From);
    if (Next == llvm::StringRef::npos)
      break;
    R += "\\";
    R += Input[Next];

    From = Next + 1;
  }
  return R;
}

/// Renders \p Input as an inline block of code in markdown. The returned value
/// is surrounded by backticks and the inner contents are properly escaped.
std::string renderInlineBlock(llvm::StringRef Input) {
  std::string R;
  // Double all backticks to make sure we don't close the inline block early.
  for (size_t From = 0; From < Input.size();) {
    size_t Next = Input.find("`", From);
    R += Input.substr(From, Next - From);
    if (Next == llvm::StringRef::npos)
      break;
    R += "``"; // double the found backtick.

    From = Next + 1;
  }
  // If results starts with a backtick, add spaces on both sides. The spaces
  // are ignored by markdown renderers.
  if (llvm::StringRef(R).startswith("`") || llvm::StringRef(R).endswith("`"))
    return "` " + std::move(R) + " `";
  // Markdown render should ignore first and last space if both are there. We
  // add an extra pair of spaces in that case to make sure we render what the
  // user intended.
  if (llvm::StringRef(R).startswith(" ") && llvm::StringRef(R).endswith(" "))
    return "` " + std::move(R) + " `";
  return "`" + std::move(R) + "`";
}

/// Get marker required for \p Input to represent a markdown codeblock. It
/// consists of at least 3 backticks(`). Although markdown also allows to use
/// tilde(~) for code blocks, they are never used.
std::string getMarkerForCodeBlock(llvm::StringRef Input) {
  // Count the maximum number of consecutive backticks in \p Input. We need to
  // start and end the code block with more.
  unsigned MaxBackticks = 0;
  unsigned Backticks = 0;
  for (char C : Input) {
    if (C == '`') {
      ++Backticks;
      continue;
    }
    MaxBackticks = std::max(MaxBackticks, Backticks);
    Backticks = 0;
  }
  MaxBackticks = std::max(Backticks, MaxBackticks);
  // Use the corresponding number of backticks to start and end a code block.
  return std::string(/*Repeat=*/std::max(3u, MaxBackticks + 1), '`');
}

// Trims the input and concatenates whitespace blocks into a single ` `.
std::string canonicalizeSpaces(std::string Input) {
  // Goes over the string and preserves only a single ` ` for any whitespace
  // chunks, the rest is moved to the end of the string and dropped in the end.
  auto WritePtr = Input.begin();
  llvm::SmallVector<llvm::StringRef, 4> Words;
  llvm::SplitString(Input, Words);
  if (Words.empty())
    return "";
  // Go over each word and add it to the string.
  for (llvm::StringRef Word : Words) {
    if (WritePtr > Input.begin())
      *WritePtr++ = ' '; // Separate from previous block.
    llvm::for_each(Word, [&WritePtr](const char C) { *WritePtr++ = C; });
  }
  // Get rid of extra spaces.
  Input.resize(WritePtr - Input.begin());
  return Input;
}

std::string renderBlocks(llvm::ArrayRef<std::unique_ptr<Block>> Children,
                         void (Block::*RenderFunc)(llvm::raw_ostream &) const) {
  std::string R;
  llvm::raw_string_ostream OS(R);
  for (auto &C : Children)
    ((*C).*RenderFunc)(OS);
  return llvm::StringRef(OS.str()).trim().str();
}

// Puts a vertical space between blocks inside a document.
class Spacer : public Block {
public:
  void renderMarkdown(llvm::raw_ostream &OS) const override { OS << '\n'; }
  void renderPlainText(llvm::raw_ostream &OS) const override { OS << '\n'; }
};

class CodeBlock : public Block {
public:
  void renderMarkdown(llvm::raw_ostream &OS) const override {
    std::string Marker = getMarkerForCodeBlock(Contents);
    // No need to pad from previous blocks, as they should end with a new line.
    OS << Marker << Language << '\n' << Contents << '\n' << Marker << '\n';
  }

  void renderPlainText(llvm::raw_ostream &OS) const override {
    // In plaintext we want one empty line before and after codeblocks.
    OS << '\n' << Contents << "\n\n";
  }

  CodeBlock(std::string Contents, std::string Language)
      : Contents(std::move(Contents)), Language(std::move(Language)) {}

private:
  std::string Contents;
  std::string Language;
};
} // namespace

std::string Block::asMarkdown() const {
  std::string R;
  llvm::raw_string_ostream OS(R);
  renderMarkdown(OS);
  return llvm::StringRef(OS.str()).trim().str();
}

std::string Block::asPlainText() const {
  std::string R;
  llvm::raw_string_ostream OS(R);
  renderPlainText(OS);
  return llvm::StringRef(OS.str()).trim().str();
}

void Paragraph::renderMarkdown(llvm::raw_ostream &OS) const {
  llvm::StringRef Sep = "";
  for (auto &C : Chunks) {
    OS << Sep;
    switch (C.Kind) {
    case Chunk::PlainText:
      OS << renderText(C.Contents);
      break;
    case Chunk::InlineCode:
      OS << renderInlineBlock(C.Contents);
      break;
    }
    Sep = " ";
  }
  // Paragraphs are translated into markdown lines, not markdown paragraphs.
  // Therefore it only has a single linebreak afterwards.
  OS << '\n';
}

void Paragraph::renderPlainText(llvm::raw_ostream &OS) const {
  llvm::StringRef Sep = "";
  for (auto &C : Chunks) {
    OS << Sep << C.Contents;
    Sep = " ";
  }
  OS << '\n';
}

Paragraph &Paragraph::appendText(std::string Text) {
  Text = canonicalizeSpaces(std::move(Text));
  if (Text.empty())
    return *this;
  Chunks.emplace_back();
  Chunk &C = Chunks.back();
  C.Contents = std::move(Text);
  C.Kind = Chunk::PlainText;
  return *this;
}

Paragraph &Paragraph::appendCode(std::string Code) {
  Code = canonicalizeSpaces(std::move(Code));
  if (Code.empty())
    return *this;
  Chunks.emplace_back();
  Chunk &C = Chunks.back();
  C.Contents = std::move(Code);
  C.Kind = Chunk::InlineCode;
  return *this;
}

Paragraph &Document::addParagraph() {
  Children.push_back(std::make_unique<Paragraph>());
  return *static_cast<Paragraph *>(Children.back().get());
}

void Document::addSpacer() { Children.push_back(std::make_unique<Spacer>()); }

void Document::addCodeBlock(std::string Code, std::string Language) {
  Children.emplace_back(
      std::make_unique<CodeBlock>(std::move(Code), std::move(Language)));
}

std::string Document::asMarkdown() const {
  return renderBlocks(Children, &Block::renderMarkdown);
}

std::string Document::asPlainText() const {
  return renderBlocks(Children, &Block::renderPlainText);
}
} // namespace markup
} // namespace clangd
} // namespace clang
