//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Markdown.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/StringSaver.h"
#include <cassert>
#include <memory>
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
  // Only the marker and spaces may appear, with at least three markers.
  const char Allowed[] = {Marker, ' '};
  return Line.find_first_not_of(StringRef(Allowed, 2)) == StringRef::npos &&
         Line.count(Marker) >= 3;
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

// A forward cursor over the lines of a paragraph. Lines are stored untrimmed;
// callers trim where they need a normalized view.
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

// A forward cursor over the characters of a string. position() and seek() let
// it interoperate with the index-based run and delimiter helpers below.
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
  size_t End = S.find_first_not_of(C, Start);
  return (End == StringRef::npos ? S.size() : End) - Start;
}

// Strips one leading and one trailing space from a code span's content when
// both are present and the content is not all spaces, per CommonMark §6.1.
static StringRef trimCodeSpan(StringRef Code) {
  if (Code.size() >= 2 && Code.front() == ' ' && Code.back() == ' ' &&
      Code.find_first_not_of(' ') != StringRef::npos)
    return Code.drop_front().drop_back();
  return Code;
}

// Treats the start and end of the string (passed as '\0') as whitespace for the
// CommonMark flanking rules.
static bool isFlankWhitespace(char C) { return C == '\0' || isSpace(C); }

// Computes whether a delimiter run can open or close emphasis, from the
// characters immediately before and after the run, per the CommonMark §6.2
// flanking rules. Before and After are '\0' at the string boundaries.
static void computeFlanking(char Before, char Marker, char After, bool &CanOpen,
                            bool &CanClose) {
  bool AfterWS = isFlankWhitespace(After);
  bool BeforeWS = isFlankWhitespace(Before);
  bool AfterPunct = isPunct(After);
  bool BeforePunct = isPunct(Before);
  bool LeftFlanking = !AfterWS && (!AfterPunct || BeforeWS || BeforePunct);
  bool RightFlanking = !BeforeWS && (!BeforePunct || AfterWS || AfterPunct);
  if (Marker == '_') {
    // Underscore does not open or close emphasis intraword.
    CanOpen = LeftFlanking && (!RightFlanking || BeforePunct);
    CanClose = RightFlanking && (!LeftFlanking || AfterPunct);
  } else {
    CanOpen = LeftFlanking;
    CanClose = RightFlanking;
  }
}

namespace {
// One piece of inline content while emphasis is being resolved. A piece is
// either a finished content node (text, code span, or a built emphasis or
// strong node) or a run of delimiter characters that may still open or close
// emphasis. Pieces form a doubly linked list through Prev/Next so matched runs
// can be spliced out without shifting the others.
struct InlinePiece {
  MDNode *Node = nullptr; // content node, or null while this is a delimiter run
  char Ch = 0;            // '*' or '_' for a delimiter run
  size_t Len = 0;         // delimiters still available in the run
  unsigned OrigLen = 0;   // original run length, for the multiple-of-three rule
  bool CanOpen = false;
  bool CanClose = false;
  int Prev = -1;
  int Next = -1;
};
} // namespace

// Parses the inline content of a single line into a sequence of inline nodes:
// inline code (`code`), emphasis (*text* or _text_), and strong (**text** or
// __text__). Emphasis is resolved with a CommonMark-style delimiter stack: a
// first pass tokenizes the line into text, code spans, and delimiter runs (each
// tagged with its flanking flags), then a second pass walks closers back to
// openers, honoring the multiple-of-three rule. Unmatched runs stay as text.
//
// TODO: This does not yet handle links, autolinks, or backslash escapes.
static ArrayRef<MDNode *> parseInline(StringRef S, BumpPtrAllocator &Arena,
                                      StringSaver &Saver) {
  SmallVector<InlinePiece> Pool;
  int Head = -1, Tail = -1;

  auto makePiece = [&]() -> int {
    Pool.emplace_back();
    return Pool.size() - 1;
  };
  auto linkAtTail = [&](int Idx) {
    Pool[Idx].Prev = Tail;
    (Tail != -1 ? Pool[Tail].Next : Head) = Idx;
    Tail = Idx;
  };
  auto appendNode = [&](MDNode *N) {
    int Idx = makePiece();
    Pool[Idx].Node = N;
    linkAtTail(Idx);
  };
  // Content nodes pass through; a leftover delimiter run becomes a TextNode of
  // its remaining characters.
  auto pieceNode = [&](int P) -> MDNode * {
    if (Pool[P].Node)
      return Pool[P].Node;
    return new (Arena)
        TextNode(Saver.save(std::string(Pool[P].Len, Pool[P].Ch)));
  };
  // Merges adjacent TextNodes so unmatched delimiters coalesce with neighboring
  // text, then copies the result into the arena.
  auto finalize = [&](SmallVectorImpl<MDNode *> &Nodes) -> ArrayRef<MDNode *> {
    SmallVector<MDNode *> Merged;
    for (MDNode *Nd : Nodes) {
      if (isa<TextNode>(Nd) && !Merged.empty() &&
          isa<TextNode>(Merged.back())) {
        StringRef Prev = cast<TextNode>(Merged.back())->Text;
        StringRef Cur = cast<TextNode>(Nd)->Text;
        Merged.back() =
            new (Arena) TextNode(Saver.save(Prev.str() + Cur.str()));
      } else {
        Merged.push_back(Nd);
      }
    }
    return allocateArray(Merged, Arena);
  };

  // Phase 1: tokenize the line into text, code spans, and delimiter runs.
  CharReader Reader(S);
  size_t TextStart = 0;
  auto flushText = [&](size_t End) {
    if (End > TextStart)
      appendNode(new (Arena) TextNode(
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
        appendNode(new (Arena) InlineCodeNode(Saver.save(Code)));
        Reader.seek(ClosePos + OpenLen);
        TextStart = Reader.position();
        continue;
      }
      // No closing run; leave the backticks as literal text.
      Reader.seek(Pos + OpenLen);
      continue;
    }

    // Delimiter run for emphasis or strong.
    if (C == '*' || C == '_') {
      size_t RunLen = countRun(S, Pos, C);
      flushText(Pos);
      char Before = Pos == 0 ? '\0' : S[Pos - 1];
      char After = Pos + RunLen < S.size() ? S[Pos + RunLen] : '\0';
      int Idx = makePiece();
      InlinePiece &D = Pool[Idx];
      D.Ch = C;
      D.Len = RunLen;
      D.OrigLen = RunLen;
      computeFlanking(Before, C, After, D.CanOpen, D.CanClose);
      linkAtTail(Idx);
      Reader.seek(Pos + RunLen);
      TextStart = Reader.position();
      continue;
    }

    Reader.advance();
  }
  flushText(S.size());

  // Phase 2: match closers back to openers. OpenersBottom records, per closer
  // kind, how far back a failed search needs to look, keyed by delimiter char,
  // run length mod 3, and whether the closer can also open.
  SmallVector<int, 12> OpenersBottom(12, -1);
  auto bucket = [](const InlinePiece &P) {
    return (P.Ch == '_' ? 6 : 0) + (P.OrigLen % 3) * 2 + (P.CanOpen ? 1 : 0);
  };

  int Current = Head;
  while (Current != -1) {
    // Advance to the next run that can close.
    while (Current != -1 &&
           !(Pool[Current].Ch && Pool[Current].CanClose && Pool[Current].Len))
      Current = Pool[Current].Next;
    if (Current == -1)
      break;
    int Closer = Current;
    int Key = bucket(Pool[Closer]);

    // Search back for the nearest matching opener.
    int Opener = Pool[Closer].Prev;
    bool Found = false;
    while (Opener != -1 && Opener != OpenersBottom[Key]) {
      InlinePiece &O = Pool[Opener];
      if (O.Ch == Pool[Closer].Ch && O.Len && O.CanOpen) {
        unsigned Sum = O.OrigLen + Pool[Closer].OrigLen;
        bool OddMatch = (O.CanClose || Pool[Closer].CanOpen) && Sum % 3 == 0 &&
                        !(O.OrigLen % 3 == 0 && Pool[Closer].OrigLen % 3 == 0);
        if (!OddMatch) {
          Found = true;
          break;
        }
      }
      Opener = Pool[Opener].Prev;
    }

    if (!Found) {
      OpenersBottom[Key] = Pool[Closer].Prev;
      // A run that cannot also open will never match anything; keep its text
      // but stop treating it as a delimiter.
      if (!Pool[Closer].CanOpen)
        Pool[Closer].CanClose = false;
      Current = Pool[Closer].Next;
      continue;
    }

    // Wrap the pieces between opener and closer, consuming one delimiter from
    // each side for emphasis or two for strong.
    unsigned Use = Pool[Opener].Len >= 2 && Pool[Closer].Len >= 2 ? 2 : 1;
    SmallVector<MDNode *> Inner;
    for (int P = Pool[Opener].Next; P != Closer; P = Pool[P].Next)
      Inner.push_back(pieceNode(P));
    Pool[Opener].Len -= Use;
    Pool[Closer].Len -= Use;
    MDNode *Emph =
        Use == 2
            ? static_cast<MDNode *>(new (Arena) StrongNode(finalize(Inner)))
            : static_cast<MDNode *>(new (Arena) EmphasisNode(finalize(Inner)));
    int EP = makePiece();
    Pool[EP].Node = Emph;
    Pool[EP].Prev = Opener;
    Pool[EP].Next = Closer;
    Pool[Opener].Next = EP;
    Pool[Closer].Prev = EP;

    // Drop the opener or closer once its run is fully consumed.
    if (Pool[Opener].Len == 0) {
      int Pr = Pool[Opener].Prev;
      Pool[EP].Prev = Pr;
      (Pr != -1 ? Pool[Pr].Next : Head) = EP;
    }
    if (Pool[Closer].Len == 0) {
      int Nx = Pool[Closer].Next;
      Pool[EP].Next = Nx;
      (Nx != -1 ? Pool[Nx].Prev : Tail) = EP;
      Current = Nx;
    } else {
      Current = Closer;
    }
  }

  // Phase 3: collect the surviving pieces, dropping fully consumed delimiters.
  SmallVector<MDNode *> Result;
  for (int P = Head; P != -1; P = Pool[P].Next)
    if (Pool[P].Node || Pool[P].Len)
      Result.push_back(pieceNode(P));
  return finalize(Result);
}

// Parses a fenced code block opened with ``` or ~~~. The cursor must be on the
// opening fence; the fence, body lines, and closing fence are consumed.
//
// TODO: Follow CommonMark spec §4.5 more closely. Opening fences may be
// indented up to 3 spaces, the closing fence must use the same character and be
// at least as long as the opening fence, and the closing fence may only be
// followed by spaces.
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
        llvm::all_of(CodeLine.take_front(3),
                     [Fence](char C) { return C == Fence; }))
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

// Splits a pipe table row into cell texts. A single optional leading and
// trailing pipe are dropped, then the remainder is split on '|' and each cell
// is trimmed.
// TODO: A '|' inside a code span or escaped as "\|" should not split a cell.
static void splitTableRow(StringRef Row, SmallVectorImpl<StringRef> &Cells) {
  Row = Row.trim();
  if (Row.starts_with("|"))
    Row = Row.drop_front();
  if (Row.ends_with("|"))
    Row = Row.drop_back();
  SmallVector<StringRef> Parts;
  Row.split(Parts, '|');
  for (StringRef Part : Parts)
    Cells.push_back(Part.trim());
}

// Parses a pipe table. The cursor must be on the header row, with a separator
// row following; consecutive lines containing a | are taken as body rows. Each
// cell's text is parsed into inline nodes.
static TableNode *parsePipeTable(LineReader &Reader, BumpPtrAllocator &Arena,
                                 StringSaver &Saver) {
  auto parseRow = [&](StringRef Line) -> TableRow {
    SmallVector<StringRef> CellTexts;
    splitTableRow(Line, CellTexts);
    SmallVector<TableCell> Cells;
    for (StringRef Text : CellTexts)
      Cells.push_back(TableCell{parseInline(Text, Arena, Saver)});
    return TableRow{allocateArray(Cells, Arena)};
  };

  TableRow Header = parseRow(Reader.advance());
  Reader.advance(); // skip the alignment separator row
  SmallVector<TableRow> Body;
  while (!Reader.atEnd() && Reader.peek().trim().contains('|'))
    Body.push_back(parseRow(Reader.advance()));
  auto *Table = new (Arena) TableNode(Header, allocateArray(Body, Arena));
  LDBG() << "emitting TableNode header_cells=" << Header.Cells.size()
         << " body_rows=" << Body.size();
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
// are consumed. Item numbers after the first are not validated. Returns nullptr
// without consuming input when the start number does not fit in unsigned, so
// the caller can fall back to treating the line as plain text.
static OrderedListNode *parseOrderedList(LineReader &Reader,
                                         BumpPtrAllocator &Arena,
                                         StringSaver &Saver) {
  unsigned Start = 0;
  if (Reader.peek().trim().take_while(isDigit).getAsInteger(10, Start))
    return nullptr;
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

    // Ordered list item: digits followed by a period and a space. A start
    // number too large for unsigned falls through to plain text.
    if (isOrderedListItem(Line)) {
      if (auto *List = parseOrderedList(Reader, Arena, Saver)) {
        Nodes.push_back(List);
        continue;
      }
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
