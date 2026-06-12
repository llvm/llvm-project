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
#include <cassert>

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

// Interns a StringRef into the arena so it outlives the parse loop.
static StringRef internString(StringRef S, BumpPtrAllocator &Arena) {
  if (S.empty())
    return {};
  char *Buf = Arena.Allocate<char>(S.size());
  std::copy(S.begin(), S.end(), Buf);
  return StringRef(Buf, S.size());
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

// Finds the start index of a closing emphasis run of exactly Count copies of C,
// searching forward from From. Requires non-whitespace immediately inside both
// the opening and closing delimiters and non-empty content, a simplified take
// on the CommonMark §6.2 flanking rules. Returns StringRef::npos if no valid
// closing run exists.
static size_t findClosingDelim(StringRef S, size_t From, char C, size_t Count) {
  size_t E = S.size();
  // Opening delimiter is not left-flanking if whitespace follows it.
  if (From >= E || isSpace(S[From]))
    return StringRef::npos;
  for (size_t J = From; J + Count <= E; ++J) {
    if (S[J] != C)
      continue;
    size_t Run = countRun(S, J, C);
    if (Run != Count) {
      J += Run - 1; // Skip the whole run; the loop's ++J lands past it.
      continue;
    }
    // Reject empty content and closing runs that are not right-flanking.
    if (J == From || isSpace(S[J - 1]))
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
static ArrayRef<MDNode *> parseInline(StringRef S, BumpPtrAllocator &Arena) {
  SmallVector<MDNode *> Nodes;
  size_t TextStart = 0, Pos = 0, E = S.size();

  auto flushText = [&](size_t End) {
    if (End > TextStart)
      Nodes.push_back(new (Arena) TextNode(
          internString(S.substr(TextStart, End - TextStart), Arena)));
  };

  while (Pos < E) {
    char C = S[Pos];

    // Inline code span: an opening backtick run closed by a run of the same
    // length.
    if (C == '`') {
      size_t OpenLen = countRun(S, Pos, '`');
      size_t ClosePos = Pos + OpenLen;
      while (ClosePos < E && countRun(S, ClosePos, '`') != OpenLen)
        ClosePos += S[ClosePos] == '`' ? countRun(S, ClosePos, '`') : 1;
      if (ClosePos < E) {
        flushText(Pos);
        StringRef Code =
            trimCodeSpan(S.substr(Pos + OpenLen, ClosePos - (Pos + OpenLen)));
        Nodes.push_back(new (Arena) InlineCodeNode(internString(Code, Arena)));
        Pos = ClosePos + OpenLen;
        TextStart = Pos;
        continue;
      }
      // No closing run; leave the backticks as literal text.
      Pos += OpenLen;
      continue;
    }

    // Emphasis (*text*, _text_) and strong (**text**, __text__).
    if (C == '*' || C == '_') {
      // Strong binds the two-delimiter form before single-delimiter emphasis.
      if (Pos + 1 < E && S[Pos + 1] == C) {
        size_t Close = findClosingDelim(S, Pos + 2, C, 2);
        if (Close != StringRef::npos) {
          flushText(Pos);
          StringRef Inner = S.substr(Pos + 2, Close - (Pos + 2));
          Nodes.push_back(new (Arena) StrongNode(parseInline(Inner, Arena)));
          Pos = Close + 2;
          TextStart = Pos;
          continue;
        }
      }
      size_t Close = findClosingDelim(S, Pos + 1, C, 1);
      if (Close != StringRef::npos) {
        flushText(Pos);
        StringRef Inner = S.substr(Pos + 1, Close - (Pos + 1));
        Nodes.push_back(new (Arena) EmphasisNode(parseInline(Inner, Arena)));
        Pos = Close + 1;
        TextStart = Pos;
        continue;
      }
    }

    ++Pos;
  }

  flushText(E);
  return allocateArray(Nodes, Arena);
}

ArrayRef<MDNode *> parseMarkdown(StringRef ParagraphText,
                                 BumpPtrAllocator &Arena) {
  if (ParagraphText.trim().empty())
    return {};

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

    // TODO: Follow CommonMark spec §4.5 more closely -- opening fences may be
    // indented up to 3 spaces, the closing fence must use the same character
    // and be at least as long as the opening fence, and the closing fence may
    // only be followed by spaces. Doxygen specifics should be handled on a
    // case-by-case basis.
    if (Line.starts_with("```") || Line.starts_with("~~~")) {
      char Fence = Line[0];
      StringRef Lang = internString(Line.drop_front(3).trim(), Arena);
      Reader.advance(); // consume opening fence
      SmallVector<StringRef> CodeLines;
      while (!Reader.atEnd()) {
        StringRef CodeLine = Reader.peek().trim();
        if (CodeLine.size() >= 3 &&
            all_of(CodeLine.take_front(3),
                   [Fence](char C) { return C == Fence; }))
          break;
        CodeLines.push_back(internString(Reader.advance(), Arena));
      }
      if (!Reader.atEnd())
        Reader.advance(); // consume closing fence
      auto *Code =
          new (Arena) FencedCodeNode(Lang, allocateArray(CodeLines, Arena));
      LDBG() << "emitting FencedCodeNode lang='" << Lang
             << "' lines=" << CodeLines.size();
      Nodes.push_back(Code);
      continue;
    }

    // Pipe table: current line has | and next line is a separator row.
    if (Line.contains('|') && isSepRow(Reader.peek(1).trim())) {
      SmallVector<StringRef> Rows;
      while (!Reader.atEnd() && Reader.peek().trim().contains('|'))
        Rows.push_back(internString(Reader.advance().trim(), Arena));
      auto *Table = new (Arena) TableNode(allocateArray(Rows, Arena));
      LDBG() << "emitting TableNode rows=" << Rows.size();
      Nodes.push_back(Table);
      continue;
    }

    // Unordered list item.
    if (isListItem(Line)) {
      SmallVector<ListItemNode *> Items;
      while (!Reader.atEnd()) {
        StringRef L = Reader.peek().trim();
        if (!isListItem(L))
          break;
        StringRef ItemText = internString(L.drop_front(2).trim(), Arena);
        SmallVector<MDNode *> ItemChildren;
        ItemChildren.push_back(new (Arena) TextNode(ItemText));
        auto *Item =
            new (Arena) ListItemNode(allocateArray(ItemChildren, Arena));
        Items.push_back(Item);
        Reader.advance();
      }
      auto *List = new (Arena) UnorderedListNode(allocateArray(Items, Arena));
      LDBG() << "emitting UnorderedListNode items=" << Items.size();
      Nodes.push_back(List);
      continue;
    }

    // Plain text, scanned for inline constructs (emphasis, strong, code).
    for (MDNode *Inline : parseInline(Line, Arena))
      Nodes.push_back(Inline);
    Reader.advance();
  }

  LDBG() << "parseMarkdown done nodes=" << Nodes.size();
  return allocateArray(Nodes, Arena);
}

} // namespace clang::doc::markdown