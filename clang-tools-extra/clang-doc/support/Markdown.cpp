//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markdown.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/StringSaver.h"
#include <cassert>
#include <string>

#define DEBUG_TYPE "clang-doc"

using namespace llvm;

namespace clang::doc::markdown {

// Allocates a contiguous array of T in the arena and returns an ArrayRef.
template <typename T>
static ArrayRef<T> allocateArray(SmallVectorImpl<T> &Vec,
                                 BumpPtrAllocator &Arena) {
  if (Vec.empty())
    return {};
  T *Allocated = Arena.Allocate<T>(Vec.size());
  std::uninitialized_copy(Vec.begin(), Vec.end(), Allocated);
  return ArrayRef<T>(Allocated, Vec.size());
}

// A line is a table separator if it only contains |, -, :, and spaces,
// and has at least one -.
static bool isSepRow(StringRef Line) {
  return Line.contains('-') &&
         Line.find_first_not_of("|-: ") == StringRef::npos;
}

// Returns true if Line begins with a bullet list marker (-, *, or +)
// followed by a space.
static bool isListItem(StringRef Line) {
  return Line.starts_with("- ") || Line.starts_with("* ") ||
         Line.starts_with("+ ");
}

// Returns true if Line begins with an ordered list marker: one or more digits
// followed by a period and a space (e.g. "1. ", "42. ").
static bool isOrderedListItem(StringRef Line) {
  size_t Dot = Line.find_first_not_of("0123456789");
  return Dot != StringRef::npos && Dot > 0 && Line[Dot] == '.' &&
         Dot + 1 < Line.size() && Line[Dot + 1] == ' ';
}

// Returns true if Line is a thematic break: three or more matching -, *, or _
// characters, optionally separated by spaces, with nothing else. Line is
// expected to be trimmed.
static bool isThematicBreak(StringRef Line) {
  char Marker = Line.empty() ? '\0' : Line[0];
  if (Marker != '-' && Marker != '*' && Marker != '_')
    return false;
  unsigned Count = 0;
  for (char C : Line) {
    if (C == Marker)
      ++Count;
    else if (C != ' ')
      return false;
  }
  return Count >= 3;
}

// Returns true if Line is a block quote line: it starts with "> ", or is a bare
// ">" marking an empty quote line.
static bool isBlockQuote(StringRef Line) {
  return Line.starts_with("> ") || Line == ">";
}

// Returns the ATX heading level (1 to 6) when Line is an ATX heading: one to
// six leading # characters followed by a space. Returns 0 otherwise, so seven
// or more # characters fall back to plain text.
static unsigned atxHeadingLevel(StringRef Line) {
  size_t Level = Line.find_first_not_of('#');
  if (Level == StringRef::npos || Level < 1 || Level > 6 || Line[Level] != ' ')
    return 0;
  return Level;
}

// A forward cursor over the lines of a paragraph. Encapsulates the parse
// position so the loop can inspect the current or an upcoming line and consume
// lines without manual index arithmetic. Lines are stored untrimmed; callers
// trim where they need a normalized view.
class LineReader {
public:
  explicit LineReader(ArrayRef<StringRef> Lines) : Lines(Lines) {}

  // True once every line has been consumed.
  bool atEnd() const { return Pos >= Lines.size(); }

  // The current line, untrimmed. Must not be called when atEnd().
  StringRef peek() const {
    assert(!atEnd() && "peek past end of input");
    return Lines[Pos];
  }

  // The line Offset positions ahead of the cursor, or an empty StringRef when
  // that position is past the end. peek(0) is the current line.
  StringRef peek(size_t Offset) const {
    size_t Target = Pos + Offset;
    return Target < Lines.size() ? Lines[Target] : StringRef();
  }

  // Consume the current line and return it, untrimmed. Must not be called when
  // atEnd().
  StringRef advance() {
    assert(!atEnd() && "advance past end of input");
    return Lines[Pos++];
  }

private:
  ArrayRef<StringRef> Lines;
  size_t Pos = 0;
};

// A forward cursor over the characters of a string. The character-level analog
// of LineReader: the inline scanner inspects the current or an upcoming
// character and consumes characters without manual index arithmetic. position()
// and seek() let it interoperate with the index-based run and delimiter helpers
// below, since inline constructs are not consumed one character at a time.
class CharReader {
public:
  explicit CharReader(StringRef S) : S(S) {}

  // True once every character has been consumed.
  bool atEnd() const { return Pos >= S.size(); }

  // The current character. Must not be called when atEnd().
  char peek() const {
    assert(!atEnd() && "peek past end of input");
    return S[Pos];
  }

  // The character Offset positions ahead of the cursor, or '\0' when that
  // position is past the end. peek(0) is the current character.
  char peek(size_t Offset) const {
    size_t Target = Pos + Offset;
    return Target < S.size() ? S[Target] : '\0';
  }

  // Consume the current character and return it. Must not be called when
  // atEnd().
  char advance() {
    assert(!atEnd() && "advance past end of input");
    return S[Pos++];
  }

  // The current scan position, for substring, run, and delimiter computations.
  size_t position() const { return Pos; }

  // Move the cursor to an absolute position, used to skip past a matched span.
  void seek(size_t NewPos) { Pos = NewPos; }

private:
  StringRef S;
  size_t Pos = 0;
};

// Returns the number of consecutive copies of C starting at S[Start].
static size_t countRun(StringRef S, size_t Start, char C) {
  size_t I = Start;
  while (I < S.size() && S[I] == C)
    ++I;
  return I - Start;
}

// Strips one leading and one trailing space from a code span's content when
// both are present and the content is not all spaces, per CommonMark §6.1.
static StringRef trimCodeSpan(StringRef Code) {
  if (Code.size() >= 2 && Code.front() == ' ' && Code.back() == ' ' &&
      Code.find_first_not_of(' ') != StringRef::npos)
    return Code.drop_front().drop_back();
  return Code;
}

// Finds the start index of a closing emphasis run of exactly DelimLen copies of
// DelimChar, searching forward from StartPos. Requires non-whitespace
// immediately inside both the opening and closing delimiters and non-empty
// content, a simplified take on the CommonMark §6.2 flanking rules. Returns
// StringRef::npos if no valid closing run exists.
static size_t findClosingDelim(StringRef S, size_t StartPos, char DelimChar,
                               size_t DelimLen) {
  size_t E = S.size();
  // Opening delimiter is not left-flanking if whitespace follows it.
  if (StartPos >= E || isSpace(S[StartPos]))
    return StringRef::npos;
  for (size_t J = StartPos; J + DelimLen <= E; ++J) {
    if (S[J] != DelimChar)
      continue;
    size_t Run = countRun(S, J, DelimChar);
    if (Run != DelimLen) {
      J += Run - 1; // Skip the whole run; the loop's ++J lands past it.
      continue;
    }
    // Reject empty content and closing runs that are not right-flanking.
    if (J == StartPos || isSpace(S[J - 1]))
      continue;
    return J;
  }
  return StringRef::npos;
}

// Parses the inline content of a single line into a sequence of inline nodes:
// inline code (`code`), strong (**text** or __text__), and emphasis (*text* or
// _text_). Runs that match no construct become TextNodes. Emphasis and strong
// recurse so their content may itself contain inline constructs. Text with no
// markers yields a single TextNode.
//
// TODO: This covers the common cases but not the full CommonMark §6 inline
// model (delimiter stacks, intraword underscore rules, links, autolinks).
static ArrayRef<MDNode *> parseInline(StringRef S, BumpPtrAllocator &Arena,
                                      StringSaver &Saver) {
  SmallVector<MDNode *> Nodes;
  CharReader Reader(S);
  size_t TextStart = 0;

  auto flushText = [&](size_t End) {
    if (End > TextStart)
      Nodes.push_back(new (Arena) TextNode(
          Saver.save(S.substr(TextStart, End - TextStart))));
  };

  while (!Reader.atEnd()) {
    size_t Pos = Reader.position();
    char C = Reader.peek();

    // Inline code span: an opening backtick run closed by a run of the same
    // length.
    if (C == '`') {
      size_t OpenLen = countRun(S, Pos, '`');
      size_t ClosePos = Pos + OpenLen;
      while (ClosePos < S.size() && countRun(S, ClosePos, '`') != OpenLen)
        ClosePos += S[ClosePos] == '`' ? countRun(S, ClosePos, '`') : 1;
      if (ClosePos < S.size()) {
        flushText(Pos);
        StringRef Code =
            trimCodeSpan(S.substr(Pos + OpenLen, ClosePos - (Pos + OpenLen)));
        Nodes.push_back(new (Arena) InlineCodeNode(Saver.save(Code)));
        Reader.seek(ClosePos + OpenLen);
        TextStart = Reader.position();
        continue;
      }
      // No closing run; leave the backticks as literal text.
      Reader.seek(Pos + OpenLen);
      continue;
    }

    // Emphasis (*text*, _text_) and strong (**text**, __text__).
    if (C == '*' || C == '_') {
      // Strong binds the two-delimiter form before single-delimiter emphasis.
      if (Reader.peek(1) == C) {
        size_t Close = findClosingDelim(S, Pos + 2, C, 2);
        if (Close != StringRef::npos) {
          flushText(Pos);
          StringRef Inner = S.substr(Pos + 2, Close - (Pos + 2));
          Nodes.push_back(new (Arena)
                              StrongNode(parseInline(Inner, Arena, Saver)));
          Reader.seek(Close + 2);
          TextStart = Reader.position();
          continue;
        }
      }
      size_t Close = findClosingDelim(S, Pos + 1, C, 1);
      if (Close != StringRef::npos) {
        flushText(Pos);
        StringRef Inner = S.substr(Pos + 1, Close - (Pos + 1));
        Nodes.push_back(new (Arena)
                            EmphasisNode(parseInline(Inner, Arena, Saver)));
        Reader.seek(Close + 1);
        TextStart = Reader.position();
        continue;
      }
    }

    Reader.advance();
  }

  flushText(S.size());
  return allocateArray(Nodes, Arena);
}

// Parses a fenced code block opened with ``` or ~~~. The cursor must be on the
// opening fence; the fence, body lines, and closing fence are consumed.
//
// TODO: Follow CommonMark spec §4.5 more closely -- opening fences may be
// indented up to 3 spaces, the closing fence must use the same character and be
// at least as long as the opening fence, and the closing fence may only be
// followed by spaces. Doxygen specifics should be handled on a case-by-case
// basis.
static FencedCodeNode *parseFencedCode(LineReader &Reader,
                                       BumpPtrAllocator &Arena,
                                       StringSaver &Saver) {
  StringRef Open = Reader.peek().trim();
  char Fence = Open[0];
  StringRef Lang = Saver.save(Open.drop_front(3).trim());
  Reader.advance(); // consume opening fence
  SmallVector<StringRef> CodeLines;
  while (!Reader.atEnd()) {
    StringRef CodeLine = Reader.peek().trim();
    if (CodeLine.size() >= 3 &&
        all_of(CodeLine.take_front(3), [Fence](char C) { return C == Fence; }))
      break;
    CodeLines.push_back(Saver.save(Reader.advance()));
  }
  if (!Reader.atEnd())
    Reader.advance(); // consume closing fence
  auto *Code =
      new (Arena) FencedCodeNode(Lang, allocateArray(CodeLines, Arena));
  LDBG() << "emitting FencedCodeNode lang='" << Lang
         << "' lines=" << CodeLines.size();
  return Code;
}

// Parses a pipe table. The cursor must be on the header row, with a separator
// row following; consecutive lines containing a | are taken as rows.
static TableNode *parsePipeTable(LineReader &Reader, BumpPtrAllocator &Arena,
                                 StringSaver &Saver) {
  SmallVector<StringRef> Rows;
  // TODO: Rows are kept as raw line text for now. Table cells may contain
  // inline content (emphasis, code spans, links), so each row may need to be
  // split on '|' and parsed further into structured cells.
  while (!Reader.atEnd() && Reader.peek().trim().contains('|'))
    Rows.push_back(Saver.save(Reader.advance().trim()));
  auto *Table = new (Arena) TableNode(allocateArray(Rows, Arena));
  LDBG() << "emitting TableNode rows=" << Rows.size();
  return Table;
}

// Parses an unordered (bullet) list. The cursor must be on the first item;
// consecutive bullet lines are consumed into list items.
static UnorderedListNode *parseUnorderedList(LineReader &Reader,
                                             BumpPtrAllocator &Arena,
                                             StringSaver &Saver) {
  SmallVector<ListItemNode *> Items;
  while (!Reader.atEnd()) {
    StringRef L = Reader.peek().trim();
    if (!isListItem(L))
      break;
    StringRef ItemText = L.drop_front(2).trim();
    auto *Item = new (Arena) ListItemNode(parseInline(ItemText, Arena, Saver));
    Items.push_back(Item);
    Reader.advance();
  }
  auto *List = new (Arena) UnorderedListNode(allocateArray(Items, Arena));
  LDBG() << "emitting UnorderedListNode items=" << Items.size();
  return List;
}

// Parses an ordered (numbered) list. The cursor must be on the first item; the
// start number is taken from that item's marker and consecutive numbered lines
// are consumed. Item numbers after the first are not validated.
static OrderedListNode *parseOrderedList(LineReader &Reader,
                                         BumpPtrAllocator &Arena,
                                         StringSaver &Saver) {
  unsigned Start = 0;
  Reader.peek().trim().take_while(isDigit).getAsInteger(10, Start);
  SmallVector<ListItemNode *> Items;
  while (!Reader.atEnd()) {
    StringRef L = Reader.peek().trim();
    if (!isOrderedListItem(L))
      break;
    // Drop the "<digits>. " marker: the digits, the period, and the space.
    StringRef ItemText =
        L.drop_front(L.find_first_not_of("0123456789") + 2).trim();
    auto *Item = new (Arena) ListItemNode(parseInline(ItemText, Arena, Saver));
    Items.push_back(Item);
    Reader.advance();
  }
  auto *List = new (Arena) OrderedListNode(Start, allocateArray(Items, Arena));
  LDBG() << "emitting OrderedListNode start=" << Start
         << " items=" << Items.size();
  return List;
}

// Parses an ATX heading: one to six leading # characters and a space, followed
// by inline content. The cursor must be on the heading line, which is consumed.
//
// TODO: CommonMark §4.2 also allows up to 3 leading spaces and an optional
// closing run of # characters; neither is handled yet.
static HeadingNode *parseHeading(LineReader &Reader, BumpPtrAllocator &Arena,
                                 StringSaver &Saver) {
  StringRef Line = Reader.peek().trim();
  unsigned Level = atxHeadingLevel(Line);
  assert(Level >= 1 && Level <= 6 && "parseHeading called on a non-heading");
  StringRef Content = Line.drop_front(Level).trim();
  Reader.advance();
  auto *Heading =
      new (Arena) HeadingNode(Level, parseInline(Content, Arena, Saver));
  LDBG() << "emitting HeadingNode level=" << Level;
  return Heading;
}

// Parses a block quote: one or more consecutive lines beginning with "> ". The
// > marker and one following space are stripped from each line, and the
// collected text is parsed recursively, so a quote's children are block-level
// nodes and nested quotes fall out naturally.
static BlockQuoteNode *parseBlockQuote(LineReader &Reader,
                                       BumpPtrAllocator &Arena) {
  std::string Inner;
  bool First = true;
  while (!Reader.atEnd()) {
    StringRef L = Reader.peek().trim();
    if (!isBlockQuote(L))
      break;
    if (!First)
      Inner += '\n';
    First = false;
    StringRef Content = L.starts_with("> ") ? L.drop_front(2) : L.drop_front(1);
    Inner.append(Content.data(), Content.size());
    Reader.advance();
  }
  ArrayRef<MDNode *> Children = parseMarkdown(Inner, Arena);
  auto *Quote = new (Arena) BlockQuoteNode(Children);
  LDBG() << "emitting BlockQuoteNode children=" << Children.size();
  return Quote;
}

ArrayRef<MDNode *> parseMarkdown(StringRef ParagraphText,
                                 BumpPtrAllocator &Arena) {
  if (ParagraphText.trim().empty())
    return {};

  StringSaver Saver(Arena);
  SmallVector<StringRef, 16> Lines;
  ParagraphText.split(Lines, '\n');

  SmallVector<MDNode *> Nodes;
  LineReader Reader(Lines);

  while (!Reader.atEnd()) {
    StringRef Line = Reader.peek().trim();

    if (Line.empty()) {
      Reader.advance();
      continue;
    }

    // Fenced code block.
    if (Line.starts_with("```") || Line.starts_with("~~~")) {
      Nodes.push_back(parseFencedCode(Reader, Arena, Saver));
      continue;
    }

    // ATX heading: 1 to 6 leading # characters and a space.
    if (atxHeadingLevel(Line)) {
      Nodes.push_back(parseHeading(Reader, Arena, Saver));
      continue;
    }

    // Thematic break: 3 or more matching -, *, or _ characters. Checked before
    // the list cases so that "* * *" and "- - -" are breaks, not list items.
    if (isThematicBreak(Line)) {
      Reader.advance();
      Nodes.push_back(new (Arena) ThematicBreakNode());
      LDBG() << "emitting ThematicBreakNode";
      continue;
    }

    // Block quote: consecutive lines beginning with "> ".
    if (isBlockQuote(Line)) {
      Nodes.push_back(parseBlockQuote(Reader, Arena));
      continue;
    }

    // Pipe table: current line has | and next line is a separator row.
    if (Line.contains('|') && isSepRow(Reader.peek(1).trim())) {
      Nodes.push_back(parsePipeTable(Reader, Arena, Saver));
      continue;
    }

    // Unordered list item.
    if (isListItem(Line)) {
      Nodes.push_back(parseUnorderedList(Reader, Arena, Saver));
      continue;
    }

    // Ordered list item: digits followed by a period and a space.
    if (isOrderedListItem(Line)) {
      Nodes.push_back(parseOrderedList(Reader, Arena, Saver));
      continue;
    }

    // Plain text line: scan for inline constructs (emphasis, strong, code) and
    // wrap the result in a paragraph.
    auto Inlines = parseInline(Line, Arena, Saver);
    Nodes.push_back(new (Arena) ParagraphNode(Inlines));
    Reader.advance();
  }

  LDBG() << "parseMarkdown done nodes=" << Nodes.size();
  return allocateArray(Nodes, Arena);
}

} // namespace clang::doc::markdown