//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MarkdownParser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::doc::markdown {

namespace {

/// A forward cursor over the lines of the input. Owns no storage; it points
/// into a caller-provided array of lines and tracks the current position.
/// Block parsers consume lines through this cursor instead of juggling a raw
/// index, which keeps the parsing helpers free of manual bookkeeping.
class LineCursor {
public:
  explicit LineCursor(llvm::ArrayRef<llvm::StringRef> Lines) : Lines(Lines) {}

  bool atEnd() const { return Pos >= Lines.size(); }

  /// The current line, trimmed of surrounding whitespace.
  llvm::StringRef peek() const { return Lines[Pos].trim(); }

  /// The current line as it appears in the source, without trimming.
  llvm::StringRef peekRaw() const { return Lines[Pos]; }

  void advance() { ++Pos; }

private:
  llvm::ArrayRef<llvm::StringRef> Lines;
  size_t Pos = 0;
};

} // namespace

static bool isListMarker(llvm::StringRef Line) {
  return Line.starts_with("- ") || Line.starts_with("* ") ||
         Line.starts_with("+ ");
}

static bool isThematicBreak(llvm::StringRef Line) {
  if (Line.empty())
    return false;
  char Marker = Line.front();
  if (Marker != '-' && Marker != '*' && Marker != '_')
    return false;
  llvm::SmallString<8> Allowed;
  Allowed += Marker;
  Allowed += ' ';
  if (Line.find_first_not_of(Allowed) != llvm::StringRef::npos)
    return false;
  return Line.count(Marker) >= 3;
}

static bool isFence(llvm::StringRef Line) {
  return Line.starts_with("```") || Line.starts_with("~~~");
}

/// Returns the heading level (1-6) if Line is an ATX heading, or 0 otherwise.
static unsigned getHeadingLevel(llvm::StringRef Line) {
  unsigned Level = Line.take_while([](char C) { return C == '#'; }).size();
  if (Level == 0 || Level > 6 || Level >= Line.size() || Line[Level] != ' ')
    return 0;
  return Level;
}

/// True if Line begins a new block, meaning paragraph accumulation must stop.
static bool startsNewBlock(llvm::StringRef Line) {
  return Line.empty() || isThematicBreak(Line) || isFence(Line) ||
         isListMarker(Line) || getHeadingLevel(Line) != 0;
}

static TextNode *makeText(llvm::StringRef Text, ASTContext &Ctx) {
  return Ctx.allocate<TextNode>(Ctx.intern(Text));
}

static BlockNode *parseThematicBreak(LineCursor &Cursor, ASTContext &Ctx) {
  Cursor.advance();
  return Ctx.allocate<ThematicBreakNode>();
}

static BlockNode *parseFencedCode(LineCursor &Cursor, ASTContext &Ctx) {
  llvm::StringRef Opening = Cursor.peek();
  char Fence = Opening.front();
  llvm::StringRef Lang = Opening.drop_front(3).trim();
  Cursor.advance();

  llvm::SmallString<256> Code;
  while (!Cursor.atEnd()) {
    llvm::StringRef Line = Cursor.peek();
    bool IsClosingFence = Line.size() >= 3 && Line[0] == Fence &&
                          Line[1] == Fence && Line[2] == Fence;
    if (IsClosingFence) {
      Cursor.advance();
      break;
    }
    if (!Code.empty())
      Code += '\n';
    Code += Cursor.peekRaw();
    Cursor.advance();
  }
  return Ctx.allocate<FencedCodeNode>(Lang, Ctx.intern(Code));
}

static BlockNode *parseHeading(LineCursor &Cursor, ASTContext &Ctx) {
  llvm::StringRef Line = Cursor.peek();
  unsigned Level = getHeadingLevel(Line);
  llvm::StringRef Content = Line.drop_front(Level + 1).trim();
  auto *Heading = Ctx.allocate<HeadingNode>(Level);
  Heading->addChild(*makeText(Content, Ctx));
  Cursor.advance();
  return Heading;
}

static BlockNode *parseUnorderedList(LineCursor &Cursor, ASTContext &Ctx) {
  auto *List = Ctx.allocate<UnorderedListNode>();
  while (!Cursor.atEnd() && isListMarker(Cursor.peek())) {
    llvm::StringRef ItemText = Cursor.peek().drop_front(2).trim();
    auto *Item = Ctx.allocate<ListItemNode>();
    Item->addChild(*makeText(ItemText, Ctx));
    List->addItem(*Item);
    Cursor.advance();
  }
  return List;
}

static BlockNode *parseParagraph(LineCursor &Cursor, ASTContext &Ctx) {
  llvm::SmallString<256> Text;
  while (!Cursor.atEnd() && !startsNewBlock(Cursor.peek())) {
    if (!Text.empty())
      Text += ' ';
    Text += Cursor.peek();
    Cursor.advance();
  }
  auto *Para = Ctx.allocate<ParagraphNode>();
  Para->addChild(*makeText(Text, Ctx));
  return Para;
}

/// Parses the block starting at the cursor's current position. The cursor is
/// guaranteed to be on a non-empty line that begins a block.
static BlockNode *parseBlock(LineCursor &Cursor, ASTContext &Ctx) {
  llvm::StringRef Line = Cursor.peek();

  // Thematic breaks are checked first: "---" and "- - -" would otherwise be
  // misread as a fence or a list marker.
  if (isThematicBreak(Line))
    return parseThematicBreak(Cursor, Ctx);
  if (isFence(Line))
    return parseFencedCode(Cursor, Ctx);
  if (getHeadingLevel(Line) != 0)
    return parseHeading(Cursor, Ctx);
  if (isListMarker(Line))
    return parseUnorderedList(Cursor, Ctx);
  return parseParagraph(Cursor, Ctx);
}

DocumentNode *parseMarkdown(llvm::StringRef Text, ASTContext &Ctx) {
  auto *Doc = Ctx.allocate<DocumentNode>();

  llvm::SmallVector<llvm::StringRef> Lines;
  Text.split(Lines, '\n');

  LineCursor Cursor(Lines);
  while (!Cursor.atEnd()) {
    if (Cursor.peek().empty()) {
      Cursor.advance();
      continue;
    }
    Doc->addChild(*parseBlock(Cursor, Ctx));
  }
  return Doc;
}

} // namespace clang::doc::markdown
