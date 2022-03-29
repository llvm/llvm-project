//===--- Format.cpp -----------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Format.h"
#include "support/Logger.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/Unicode.h"

namespace clang {
namespace clangd {
namespace {

/// Append closing brackets )]} to \p Code to make it well-formed.
/// Clang-format conservatively refuses to format files with unmatched brackets
/// as it isn't sure where the errors are and so can't correct.
/// When editing, it's reasonable to assume code before the cursor is complete.
void closeBrackets(std::string &Code, const format::FormatStyle &Style) {
  SourceManagerForFile FileSM("mock_file.cpp", Code);
  auto &SM = FileSM.get();
  FileID FID = SM.getMainFileID();
  LangOptions LangOpts = format::getFormattingLangOpts(Style);
  Lexer Lex(FID, SM.getBufferOrFake(FID), SM, LangOpts);
  Token Tok;
  std::vector<char> Brackets;
  while (!Lex.LexFromRawLexer(Tok)) {
    switch(Tok.getKind()) {
      case tok::l_paren:
        Brackets.push_back(')');
        break;
      case tok::l_brace:
        Brackets.push_back('}');
        break;
      case tok::l_square:
        Brackets.push_back(']');
        break;
      case tok::r_paren:
        if (!Brackets.empty() && Brackets.back() == ')')
          Brackets.pop_back();
        break;
      case tok::r_brace:
        if (!Brackets.empty() && Brackets.back() == '}')
          Brackets.pop_back();
        break;
      case tok::r_square:
        if (!Brackets.empty() && Brackets.back() == ']')
          Brackets.pop_back();
        break;
      default:
        continue;
    }
  }
  // Attempt to end any open comments first.
  Code.append("\n// */\n");
  Code.append(Brackets.rbegin(), Brackets.rend());
}

static StringRef commentMarker(llvm::StringRef Line) {
  for (StringRef Marker : {"///", "//"}){
    auto I = Line.rfind(Marker);
    if (I != StringRef::npos)
      return Line.substr(I, Marker.size());
  }
  return "";
}

llvm::StringRef firstLine(llvm::StringRef Code) {
  return Code.take_until([](char C) { return C == '\n'; });
}

llvm::StringRef lastLine(llvm::StringRef Code) {
  llvm::StringRef Rest = Code;
  while (!Rest.empty() && Rest.back() != '\n')
    Rest = Rest.drop_back();
  return Code.substr(Rest.size());
}

// Filename is needed for tooling::Replacement and some overloads of reformat().
// Its value should not affect the outcome. We use the default from reformat().
llvm::StringRef Filename = "<stdin>";

// tooling::Replacement from overlapping StringRefs: From must be part of Code.
tooling::Replacement replacement(llvm::StringRef Code, llvm::StringRef From,
                                 llvm::StringRef To) {
  assert(From.begin() >= Code.begin() && From.end() <= Code.end());
  // The filename is required but ignored.
  return tooling::Replacement(Filename, From.data() - Code.data(),
                              From.size(), To);
}

// High-level representation of incremental formatting changes.
// The changes are made in two steps.
// 1) a (possibly-empty) set of changes synthesized by clangd (e.g. adding
//    comment markers when splitting a line comment with a newline).
// 2) a selective clang-format run:
//    - the "source code" passed to clang format is the code up to the cursor,
//      a placeholder for the cursor, and some closing brackets
//    - the formatting is restricted to the cursor and (possibly) other ranges
//      (e.g. the old line when inserting a newline).
//    - changes before the cursor are applied, those after are discarded.
struct IncrementalChanges {
  // Changes that should be applied before running clang-format.
  tooling::Replacements Changes;
  // Ranges of the original source code that should be clang-formatted.
  // The CursorProxyText will also be formatted.
  std::vector<tooling::Range> FormatRanges;
  // The source code that should stand in for the cursor when clang-formatting.
  // e.g. after inserting a newline, a line-comment at the cursor is used to
  // ensure that the newline is preserved.
  std::string CursorPlaceholder;
};

// The two functions below, columnWidth() and columnWidthWithTabs(), were
// adapted from similar functions in clang/lib/Format/Encoding.h.
// FIXME: Move those functions to clang/include/clang/Format.h and reuse them?

// Helper function for columnWidthWithTabs().
inline unsigned columnWidth(StringRef Text) {
  int ContentWidth = llvm::sys::unicode::columnWidthUTF8(Text);
  if (ContentWidth < 0)
    return Text.size(); // fallback for unprintable characters
  return ContentWidth;
}

// Returns the number of columns required to display the \p Text on a terminal
// with the \p TabWidth.
inline unsigned columnWidthWithTabs(StringRef Text, unsigned TabWidth) {
  unsigned TotalWidth = 0;
  StringRef Tail = Text;
  for (;;) {
    StringRef::size_type TabPos = Tail.find('\t');
    if (TabPos == StringRef::npos)
      return TotalWidth + columnWidth(Tail);
    TotalWidth += columnWidth(Tail.substr(0, TabPos));
    if (TabWidth)
      TotalWidth += TabWidth - TotalWidth % TabWidth;
    Tail = Tail.substr(TabPos + 1);
  }
}

// After a newline:
//  - we continue any line-comment that was split
//  - we format the old line in addition to the cursor
//  - we represent the cursor with a line comment to preserve the newline
IncrementalChanges getIncrementalChangesAfterNewline(llvm::StringRef Code,
                                                     unsigned Cursor,
                                                     unsigned TabWidth) {
  IncrementalChanges Result;
  // Before newline, code looked like:
  //    leading^trailing
  // After newline, code looks like:
  //    leading
  //    indentation^trailing
  // Where indentation was added by the editor.
  StringRef Trailing = firstLine(Code.substr(Cursor));
  StringRef Indentation = lastLine(Code.take_front(Cursor));
  if (Indentation.data() == Code.data()) {
    vlog("Typed a newline, but we're still on the first line!");
    return Result;
  }
  StringRef Leading =
      lastLine(Code.take_front(Indentation.data() - Code.data() - 1));
  StringRef NextLine = firstLine(Code.substr(Cursor + Trailing.size() + 1));

  // Strip leading whitespace on trailing line.
  StringRef TrailingTrim = Trailing.ltrim();
  if (unsigned TrailWS = Trailing.size() - TrailingTrim.size())
    cantFail(Result.Changes.add(
        replacement(Code, StringRef(Trailing.begin(), TrailWS), "")));

  // If we split a comment, replace indentation with a comment marker.
  // If the editor made the new line a comment, also respect that.
  StringRef CommentMarker = commentMarker(Leading);
  bool NewLineIsComment = !commentMarker(Indentation).empty();
  if (!CommentMarker.empty() &&
      (NewLineIsComment || !commentMarker(NextLine).empty() ||
       (!TrailingTrim.empty() && !TrailingTrim.startswith("//")))) {
    // We indent the new comment to match the previous one.
    StringRef PreComment =
        Leading.take_front(CommentMarker.data() - Leading.data());
    std::string IndentAndComment =
        (std::string(columnWidthWithTabs(PreComment, TabWidth), ' ') +
         CommentMarker + " ")
            .str();
    cantFail(
        Result.Changes.add(replacement(Code, Indentation, IndentAndComment)));
  } else {
    // Remove any indentation and let clang-format re-add it.
    // This prevents the cursor marker dragging e.g. an aligned comment with it.
    cantFail(Result.Changes.add(replacement(Code, Indentation, "")));
  }

  // If we put a the newline inside a {} pair, put } on its own line...
  if (CommentMarker.empty() && Leading.endswith("{") &&
      Trailing.startswith("}")) {
    cantFail(
        Result.Changes.add(replacement(Code, Trailing.take_front(1), "\n}")));
    // ...and format it.
    Result.FormatRanges.push_back(
        tooling::Range(Trailing.data() - Code.data() + 1, 1));
  }

  // Format the whole leading line.
  Result.FormatRanges.push_back(
      tooling::Range(Leading.data() - Code.data(), Leading.size()));

  // We use a comment to represent the cursor, to preserve the newline.
  // A trailing identifier improves parsing of e.g. for without braces.
  // Exception: if the previous line has a trailing comment, we can't use one
  // as the cursor (they will be aligned). But in this case we don't need to.
  Result.CursorPlaceholder = !CommentMarker.empty() ? "ident" : "//==\nident";

  return Result;
}

IncrementalChanges getIncrementalChanges(llvm::StringRef Code, unsigned Cursor,
                                         llvm::StringRef InsertedText,
                                         unsigned TabWidth) {
  IncrementalChanges Result;
  if (InsertedText == "\n")
    return getIncrementalChangesAfterNewline(Code, Cursor, TabWidth);

  Result.CursorPlaceholder = " /**/";
  return Result;
}

// Returns equivalent replacements that preserve the correspondence between
// OldCursor and NewCursor. If OldCursor lies in a replaced region, that
// replacement will be split.
std::vector<tooling::Replacement>
split(const tooling::Replacements &Replacements, unsigned OldCursor,
      unsigned NewCursor) {
  std::vector<tooling::Replacement> Result;
  int LengthChange = 0;
  for (const tooling::Replacement &R : Replacements) {
    if (R.getOffset() + R.getLength() <= OldCursor) { // before cursor
      Result.push_back(R);
      LengthChange += R.getReplacementText().size() - R.getLength();
    } else if (R.getOffset() < OldCursor) { // overlaps cursor
      int ReplacementSplit = NewCursor - LengthChange - R.getOffset();
      assert(ReplacementSplit >= 0 &&
             ReplacementSplit <= int(R.getReplacementText().size()) &&
             "NewCursor incompatible with OldCursor!");
      Result.push_back(tooling::Replacement(
          R.getFilePath(), R.getOffset(), OldCursor - R.getOffset(),
          R.getReplacementText().take_front(ReplacementSplit)));
      Result.push_back(tooling::Replacement(
          R.getFilePath(), OldCursor,
          R.getLength() - (OldCursor - R.getOffset()),
          R.getReplacementText().drop_front(ReplacementSplit)));
    } else if (R.getOffset() >= OldCursor) { // after cursor
      Result.push_back(R);
    }
  }
  return Result;
}

} // namespace

// We're simulating the following sequence of changes:
//   - apply the pre-formatting edits (see getIncrementalChanges)
//   - insert a placeholder for the cursor
//   - format some of the resulting code
//   - remove the cursor placeholder again
// The replacements we return are produced by composing these.
//
// The text we actually pass to clang-format is slightly different from this,
// e.g. we have to close brackets. We ensure these differences are *after*
// all the regions we want to format, and discard changes in them.
std::vector<tooling::Replacement>
formatIncremental(llvm::StringRef OriginalCode, unsigned OriginalCursor,
                  llvm::StringRef InsertedText, format::FormatStyle Style) {
  IncrementalChanges Incremental = getIncrementalChanges(
      OriginalCode, OriginalCursor, InsertedText, Style.TabWidth);
  // Never *remove* lines in response to pressing enter! This annoys users.
  if (InsertedText == "\n") {
    Style.MaxEmptyLinesToKeep = 1000;
    Style.KeepEmptyLinesAtTheStartOfBlocks = true;
  }

  // Compute the code we want to format:
  // 1) Start with code after the pre-formatting edits.
  std::string CodeToFormat = cantFail(
      tooling::applyAllReplacements(OriginalCode, Incremental.Changes));
  unsigned Cursor = Incremental.Changes.getShiftedCodePosition(OriginalCursor);
  // 2) Truncate code after the last interesting range.
  unsigned FormatLimit = Cursor;
  for (tooling::Range &R : Incremental.FormatRanges)
    FormatLimit = std::max(FormatLimit, R.getOffset() + R.getLength());
  CodeToFormat.resize(FormatLimit);
  // 3) Insert a placeholder for the cursor.
  CodeToFormat.insert(Cursor, Incremental.CursorPlaceholder);
  // 4) Append brackets after FormatLimit so the code is well-formed.
  closeBrackets(CodeToFormat, Style);

  // Determine the ranges to format:
  std::vector<tooling::Range> RangesToFormat = Incremental.FormatRanges;
  // Ranges after the cursor need to be adjusted for the placeholder.
  for (auto &R : RangesToFormat) {
    if (R.getOffset() > Cursor)
      R = tooling::Range(R.getOffset() + Incremental.CursorPlaceholder.size(),
                         R.getLength());
  }
  // We also format the cursor.
  RangesToFormat.push_back(
      tooling::Range(Cursor, Incremental.CursorPlaceholder.size()));
  // Also update FormatLimit for the placeholder, we'll use this later.
  FormatLimit += Incremental.CursorPlaceholder.size();

  // Run clang-format, and truncate changes at FormatLimit.
  tooling::Replacements FormattingChanges;
  format::FormattingAttemptStatus Status;
  for (const tooling::Replacement &R : format::reformat(
           Style, CodeToFormat, RangesToFormat, Filename, &Status)) {
    if (R.getOffset() + R.getLength() <= FormatLimit) // Before limit.
      cantFail(FormattingChanges.add(R));
    else if(R.getOffset() < FormatLimit) { // Overlaps limit.
      if (R.getReplacementText().empty()) // Deletions are easy to handle.
        cantFail(FormattingChanges.add(tooling::Replacement(Filename,
            R.getOffset(), FormatLimit - R.getOffset(), "")));
      else
        // Hopefully won't happen in practice?
        elog("Incremental clang-format edit overlapping cursor @ {0}!\n{1}",
             Cursor, CodeToFormat);
    }
  }
  if (!Status.FormatComplete)
    vlog("Incremental format incomplete at line {0}", Status.Line);

  // Now we are ready to compose the changes relative to OriginalCode.
  //   edits -> insert placeholder -> format -> remove placeholder.
  // We must express insert/remove as Replacements.
  tooling::Replacements InsertCursorPlaceholder(
      tooling::Replacement(Filename, Cursor, 0, Incremental.CursorPlaceholder));
  unsigned FormattedCursorStart =
               FormattingChanges.getShiftedCodePosition(Cursor),
           FormattedCursorEnd = FormattingChanges.getShiftedCodePosition(
               Cursor + Incremental.CursorPlaceholder.size());
  tooling::Replacements RemoveCursorPlaceholder(
      tooling::Replacement(Filename, FormattedCursorStart,
                           FormattedCursorEnd - FormattedCursorStart, ""));

  // We can't simply merge() and return: tooling::Replacements will combine
  // adjacent edits left and right of the cursor. This gives the right source
  // code, but loses information about where the cursor is!
  // Fortunately, none of the individual passes lose information, so:
  //  - we use merge() to compute the final Replacements
  //  - we chain getShiftedCodePosition() to compute final cursor position
  //  - we split the final Replacements at the cursor position, so that
  //    each Replacement lies either before or after the cursor.
  tooling::Replacements Final;
  unsigned FinalCursor = OriginalCursor;
#ifndef NDEBUG
  std::string FinalCode = std::string(OriginalCode);
  dlog("Initial code: {0}", FinalCode);
#endif
  for (auto Pass :
       std::vector<std::pair<const char *, const tooling::Replacements *>>{
           {"Pre-formatting changes", &Incremental.Changes},
           {"Insert placeholder", &InsertCursorPlaceholder},
           {"clang-format", &FormattingChanges},
           {"Remove placeholder", &RemoveCursorPlaceholder}}) {
    Final = Final.merge(*Pass.second);
    FinalCursor = Pass.second->getShiftedCodePosition(FinalCursor);
#ifndef NDEBUG
    FinalCode =
        cantFail(tooling::applyAllReplacements(FinalCode, *Pass.second));
    dlog("After {0}:\n{1}^{2}", Pass.first,
         StringRef(FinalCode).take_front(FinalCursor),
         StringRef(FinalCode).drop_front(FinalCursor));
#endif
  }
  return split(Final, OriginalCursor, FinalCursor);
}

unsigned
transformCursorPosition(unsigned Offset,
                        const std::vector<tooling::Replacement> &Replacements) {
  unsigned OriginalOffset = Offset;
  for (const auto &R : Replacements) {
    if (R.getOffset() + R.getLength() <= OriginalOffset) {
      // Replacement is before cursor.
      Offset += R.getReplacementText().size();
      Offset -= R.getLength();
    } else if (R.getOffset() < OriginalOffset) {
      // Replacement overlaps cursor.
      // Preserve position within replacement text, as far as possible.
      unsigned PositionWithinReplacement = Offset - R.getOffset();
      if (PositionWithinReplacement > R.getReplacementText().size()) {
        Offset += R.getReplacementText().size();
        Offset -= PositionWithinReplacement;
      }
    } else {
      // Replacement after cursor.
      break; // Replacements are sorted, the rest are also after the cursor.
    }
  }
  return Offset;
}

} // namespace clangd
} // namespace clang
