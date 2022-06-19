//===--- Markup.cpp -----------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "support/Markup.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace markup {
namespace {

// Is <contents a plausible start to an HTML tag?
// Contents may not be the rest of the line, but it's the rest of the plain
// text, so we expect to see at least the tag name.
bool looksLikeTag(llvm::StringRef Contents) {
  if (Contents.empty())
    return false;
  if (Contents.front() == '!' || Contents.front() == '?' ||
      Contents.front() == '/')
    return true;
  // Check the start of the tag name.
  if (!llvm::isAlpha(Contents.front()))
    return false;
  // Drop rest of the tag name, and following whitespace.
  Contents = Contents
                 .drop_while([](char C) {
                   return llvm::isAlnum(C) || C == '-' || C == '_' || C == ':';
                 })
                 .drop_while(llvm::isSpace);
  // The rest of the tag consists of attributes, which have restrictive names.
  // If we hit '=', all bets are off (attribute values can contain anything).
  for (; !Contents.empty(); Contents = Contents.drop_front()) {
    if (llvm::isAlnum(Contents.front()) || llvm::isSpace(Contents.front()))
      continue;
    if (Contents.front() == '>' || Contents.startswith("/>"))
      return true; // May close the tag.
    if (Contents.front() == '=')
      return true; // Don't try to parse attribute values.
    return false;  // Random punctuation means this isn't a tag.
  }
  return true; // Potentially incomplete tag.
}

// Tests whether C should be backslash-escaped in markdown.
// The string being escaped is Before + C + After. This is part of a paragraph.
// StartsLine indicates whether `Before` is the start of the line.
// After may not be everything until the end of the line.
//
// It's always safe to escape punctuation, but want minimal escaping.
// The strategy is to escape the first character of anything that might start
// a markdown grammar construct.
bool needsLeadingEscape(char C, llvm::StringRef Before, llvm::StringRef After,
                        bool StartsLine) {
  assert(Before.take_while(llvm::isSpace).empty());
  auto RulerLength = [&]() -> /*Length*/ unsigned {
    if (!StartsLine || !Before.empty())
      return false;
    llvm::StringRef A = After.rtrim();
    return llvm::all_of(A, [C](char D) { return C == D; }) ? 1 + A.size() : 0;
  };
  auto IsBullet = [&]() {
    return StartsLine && Before.empty() &&
           (After.empty() || After.startswith(" "));
  };
  auto SpaceSurrounds = [&]() {
    return (After.empty() || llvm::isSpace(After.front())) &&
           (Before.empty() || llvm::isSpace(Before.back()));
  };
  auto WordSurrounds = [&]() {
    return (!After.empty() && llvm::isAlnum(After.front())) &&
           (!Before.empty() && llvm::isAlnum(Before.back()));
  };

  switch (C) {
  case '\\': // Escaped character.
    return true;
  case '`': // Code block or inline code
    // Any number of backticks can delimit an inline code block that can end
    // anywhere (including on another line). We must escape them all.
    return true;
  case '~': // Code block
    return StartsLine && Before.empty() && After.startswith("~~");
  case '#': { // ATX heading.
    if (!StartsLine || !Before.empty())
      return false;
    llvm::StringRef Rest = After.ltrim(C);
    return Rest.empty() || Rest.startswith(" ");
  }
  case ']': // Link or link reference.
    // We escape ] rather than [ here, because it's more constrained:
    //   ](...) is an in-line link
    //   ]: is a link reference
    // The following are only links if the link reference exists:
    //   ] by itself is a shortcut link
    //   ][...] is an out-of-line link
    // Because we never emit link references, we don't need to handle these.
    return After.startswith(":") || After.startswith("(");
  case '=': // Setex heading.
    return RulerLength() > 0;
  case '_': // Horizontal ruler or matched delimiter.
    if (RulerLength() >= 3)
      return true;
    // Not a delimiter if surrounded by space, or inside a word.
    // (The rules at word boundaries are subtle).
    return !(SpaceSurrounds() || WordSurrounds());
  case '-': // Setex heading, horizontal ruler, or bullet.
    if (RulerLength() > 0)
      return true;
    return IsBullet();
  case '+': // Bullet list.
    return IsBullet();
  case '*': // Bullet list, horizontal ruler, or delimiter.
    return IsBullet() || RulerLength() >= 3 || !SpaceSurrounds();
  case '<': // HTML tag (or autolink, which we choose not to escape)
    return looksLikeTag(After);
  case '>': // Quote marker. Needs escaping at start of line.
    return StartsLine && Before.empty();
  case '&': { // HTML entity reference
    auto End = After.find(';');
    if (End == llvm::StringRef::npos)
      return false;
    llvm::StringRef Content = After.substr(0, End);
    if (Content.consume_front("#")) {
      if (Content.consume_front("x") || Content.consume_front("X"))
        return llvm::all_of(Content, llvm::isHexDigit);
      return llvm::all_of(Content, llvm::isDigit);
    }
    return llvm::all_of(Content, llvm::isAlpha);
  }
  case '.': // Numbered list indicator. Escape 12. -> 12\. at start of line.
  case ')':
    return StartsLine && !Before.empty() &&
           llvm::all_of(Before, llvm::isDigit) && After.startswith(" ");
  default:
    return false;
  }
}

/// Escape a markdown text block. Ensures the punctuation will not introduce
/// any of the markdown constructs.
std::string renderText(llvm::StringRef Input, bool StartsLine) {
  std::string R;
  for (unsigned I = 0; I < Input.size(); ++I) {
    if (needsLeadingEscape(Input[I], Input.substr(0, I), Input.substr(I + 1),
                           StartsLine))
      R.push_back('\\');
    R.push_back(Input[I]);
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
std::string canonicalizeSpaces(llvm::StringRef Input) {
  llvm::SmallVector<llvm::StringRef> Words;
  llvm::SplitString(Input, Words);
  return llvm::join(Words, " ");
}

std::string renderBlocks(llvm::ArrayRef<std::unique_ptr<Block>> Children,
                         void (Block::*RenderFunc)(llvm::raw_ostream &) const) {
  std::string R;
  llvm::raw_string_ostream OS(R);

  // Trim rulers.
  Children = Children.drop_while(
      [](const std::unique_ptr<Block> &C) { return C->isRuler(); });
  auto Last = llvm::find_if(
      llvm::reverse(Children),
      [](const std::unique_ptr<Block> &C) { return !C->isRuler(); });
  Children = Children.drop_back(Children.end() - Last.base());

  bool LastBlockWasRuler = true;
  for (const auto &C : Children) {
    if (C->isRuler() && LastBlockWasRuler)
      continue;
    LastBlockWasRuler = C->isRuler();
    ((*C).*RenderFunc)(OS);
  }

  // Get rid of redundant empty lines introduced in plaintext while imitating
  // padding in markdown.
  std::string AdjustedResult;
  llvm::StringRef TrimmedText(OS.str());
  TrimmedText = TrimmedText.trim();

  llvm::copy_if(TrimmedText, std::back_inserter(AdjustedResult),
                [&TrimmedText](const char &C) {
                  return !llvm::StringRef(TrimmedText.data(),
                                          &C - TrimmedText.data() + 1)
                              // We allow at most two newlines.
                              .endswith("\n\n\n");
                });

  return AdjustedResult;
}

// Separates two blocks with extra spacing. Note that it might render strangely
// in vscode if the trailing block is a codeblock, see
// https://github.com/microsoft/vscode/issues/88416 for details.
class Ruler : public Block {
public:
  void renderMarkdown(llvm::raw_ostream &OS) const override {
    // Note that we need an extra new line before the ruler, otherwise we might
    // make previous block a title instead of introducing a ruler.
    OS << "\n---\n";
  }
  void renderPlainText(llvm::raw_ostream &OS) const override { OS << '\n'; }
  std::unique_ptr<Block> clone() const override {
    return std::make_unique<Ruler>(*this);
  }
  bool isRuler() const override { return true; }
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

  std::unique_ptr<Block> clone() const override {
    return std::make_unique<CodeBlock>(*this);
  }

  CodeBlock(std::string Contents, std::string Language)
      : Contents(std::move(Contents)), Language(std::move(Language)) {}

private:
  std::string Contents;
  std::string Language;
};

// Inserts two spaces after each `\n` to indent each line. First line is not
// indented.
std::string indentLines(llvm::StringRef Input) {
  assert(!Input.endswith("\n") && "Input should've been trimmed.");
  std::string IndentedR;
  // We'll add 2 spaces after each new line.
  IndentedR.reserve(Input.size() + Input.count('\n') * 2);
  for (char C : Input) {
    IndentedR += C;
    if (C == '\n')
      IndentedR.append("  ");
  }
  return IndentedR;
}

class Heading : public Paragraph {
public:
  Heading(size_t Level) : Level(Level) {}
  void renderMarkdown(llvm::raw_ostream &OS) const override {
    OS << std::string(Level, '#') << ' ';
    Paragraph::renderMarkdown(OS);
  }

private:
  size_t Level;
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
  bool NeedsSpace = false;
  bool HasChunks = false;
  for (auto &C : Chunks) {
    if (C.SpaceBefore || NeedsSpace)
      OS << " ";
    switch (C.Kind) {
    case Chunk::PlainText:
      OS << renderText(C.Contents, !HasChunks);
      break;
    case Chunk::InlineCode:
      OS << renderInlineBlock(C.Contents);
      break;
    }
    HasChunks = true;
    NeedsSpace = C.SpaceAfter;
  }
  // Paragraphs are translated into markdown lines, not markdown paragraphs.
  // Therefore it only has a single linebreak afterwards.
  // VSCode requires two spaces at the end of line to start a new one.
  OS << "  \n";
}

std::unique_ptr<Block> Paragraph::clone() const {
  return std::make_unique<Paragraph>(*this);
}

/// Choose a marker to delimit `Text` from a prioritized list of options.
/// This is more readable than escaping for plain-text.
llvm::StringRef chooseMarker(llvm::ArrayRef<llvm::StringRef> Options,
                             llvm::StringRef Text) {
  // Prefer a delimiter whose characters don't appear in the text.
  for (llvm::StringRef S : Options)
    if (Text.find_first_of(S) == llvm::StringRef::npos)
      return S;
  return Options.front();
}

void Paragraph::renderPlainText(llvm::raw_ostream &OS) const {
  bool NeedsSpace = false;
  for (auto &C : Chunks) {
    if (C.SpaceBefore || NeedsSpace)
      OS << " ";
    llvm::StringRef Marker = "";
    if (C.Preserve && C.Kind == Chunk::InlineCode)
      Marker = chooseMarker({"`", "'", "\""}, C.Contents);
    OS << Marker << C.Contents << Marker;
    NeedsSpace = C.SpaceAfter;
  }
  OS << '\n';
}

void BulletList::renderMarkdown(llvm::raw_ostream &OS) const {
  for (auto &D : Items) {
    // Instead of doing this we might prefer passing Indent to children to get
    // rid of the copies, if it turns out to be a bottleneck.
    OS << "- " << indentLines(D.asMarkdown()) << '\n';
  }
  // We need a new line after list to terminate it in markdown.
  OS << '\n';
}

void BulletList::renderPlainText(llvm::raw_ostream &OS) const {
  for (auto &D : Items) {
    // Instead of doing this we might prefer passing Indent to children to get
    // rid of the copies, if it turns out to be a bottleneck.
    OS << "- " << indentLines(D.asPlainText()) << '\n';
  }
}

Paragraph &Paragraph::appendSpace() {
  if (!Chunks.empty())
    Chunks.back().SpaceAfter = true;
  return *this;
}

Paragraph &Paragraph::appendText(llvm::StringRef Text) {
  std::string Norm = canonicalizeSpaces(Text);
  if (Norm.empty())
    return *this;
  Chunks.emplace_back();
  Chunk &C = Chunks.back();
  C.Contents = std::move(Norm);
  C.Kind = Chunk::PlainText;
  C.SpaceBefore = llvm::isSpace(Text.front());
  C.SpaceAfter = llvm::isSpace(Text.back());
  return *this;
}

Paragraph &Paragraph::appendCode(llvm::StringRef Code, bool Preserve) {
  bool AdjacentCode =
      !Chunks.empty() && Chunks.back().Kind == Chunk::InlineCode;
  std::string Norm = canonicalizeSpaces(std::move(Code));
  if (Norm.empty())
    return *this;
  Chunks.emplace_back();
  Chunk &C = Chunks.back();
  C.Contents = std::move(Norm);
  C.Kind = Chunk::InlineCode;
  C.Preserve = Preserve;
  // Disallow adjacent code spans without spaces, markdown can't render them.
  C.SpaceBefore = AdjacentCode;
  return *this;
}

std::unique_ptr<Block> BulletList::clone() const {
  return std::make_unique<BulletList>(*this);
}

class Document &BulletList::addItem() {
  Items.emplace_back();
  return Items.back();
}

Document &Document::operator=(const Document &Other) {
  Children.clear();
  for (const auto &C : Other.Children)
    Children.push_back(C->clone());
  return *this;
}

void Document::append(Document Other) {
  std::move(Other.Children.begin(), Other.Children.end(),
            std::back_inserter(Children));
}

Paragraph &Document::addParagraph() {
  Children.push_back(std::make_unique<Paragraph>());
  return *static_cast<Paragraph *>(Children.back().get());
}

void Document::addRuler() { Children.push_back(std::make_unique<Ruler>()); }

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

BulletList &Document::addBulletList() {
  Children.emplace_back(std::make_unique<BulletList>());
  return *static_cast<BulletList *>(Children.back().get());
}

Paragraph &Document::addHeading(size_t Level) {
  assert(Level > 0);
  Children.emplace_back(std::make_unique<Heading>(Level));
  return *static_cast<Paragraph *>(Children.back().get());
}
} // namespace markup
} // namespace clangd
} // namespace clang
