//===- unittests/AST/CommentParser.cpp ------ Comment parser tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentParser.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentLexer.h"
#include "clang/AST/CommentSema.h"
#include "clang/Basic/CommentOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace clang {
namespace comments {

namespace {

const bool MY_DEBUG = true;

class CommentParserTest : public ::testing::Test {
protected:
  CommentParserTest()
    : FileMgr(FileMgrOpts),
      DiagID(new DiagnosticIDs()),
      Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
      SourceMgr(Diags, FileMgr),
      Traits(Allocator, CommentOptions()) {
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  llvm::BumpPtrAllocator Allocator;
  CommandTraits Traits;

  FullComment *parseString(const char *Source);
};

FullComment *CommentParserTest::parseString(const char *Source) {
  std::unique_ptr<MemoryBuffer> Buf = MemoryBuffer::getMemBuffer(Source);
  FileID File = SourceMgr.createFileID(std::move(Buf));
  SourceLocation Begin = SourceMgr.getLocForStartOfFile(File);

  Lexer L(Allocator, Diags, Traits, Begin, Source, Source + strlen(Source));

  Sema S(Allocator, SourceMgr, Diags, Traits, /*PP=*/ nullptr);
  Parser P(L, S, Allocator, SourceMgr, Diags, Traits);
  FullComment *FC = P.parseFullComment();

  if (MY_DEBUG) {
    llvm::errs() << "=== Source:\n" << Source << "\n=== AST:\n";
    FC->dump();
  }

  Token Tok;
  L.lex(Tok);
  if (Tok.is(tok::eof))
    return FC;
  else
    return nullptr;
}

::testing::AssertionResult HasChildCount(const Comment *C, size_t Count) {
  if (!C)
    return ::testing::AssertionFailure() << "Comment is NULL";

  if (Count != C->child_count())
    return ::testing::AssertionFailure()
        << "Count = " << Count
        << ", child_count = " << C->child_count();

  return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult GetChildAt(const Comment *C,
                                      size_t Idx,
                                      T *&Child) {
  if (!C)
    return ::testing::AssertionFailure() << "Comment is NULL";

  if (Idx >= C->child_count())
    return ::testing::AssertionFailure()
        << "Idx out of range.  Idx = " << Idx
        << ", child_count = " << C->child_count();

  Comment::child_iterator I = C->child_begin() + Idx;
  Comment *CommentChild = *I;
  if (!CommentChild)
    return ::testing::AssertionFailure() << "Child is NULL";

  Child = dyn_cast<T>(CommentChild);
  if (!Child)
    return ::testing::AssertionFailure()
        << "Child is not of requested type, but a "
        << CommentChild->getCommentKindName();

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasTextAt(const Comment *C,
                                     size_t Idx,
                                     StringRef Text) {
  TextComment *TC;
  ::testing::AssertionResult AR = GetChildAt(C, Idx, TC);
  if (!AR)
    return AR;

  StringRef ActualText = TC->getText();
  if (ActualText != Text)
    return ::testing::AssertionFailure()
        << "TextComment has text \"" << ActualText.str() << "\", "
           "expected \"" << Text.str() << "\"";

  if (TC->hasTrailingNewline())
    return ::testing::AssertionFailure()
        << "TextComment has a trailing newline";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasTextWithNewlineAt(const Comment *C,
                                                size_t Idx,
                                                StringRef Text) {
  TextComment *TC;
  ::testing::AssertionResult AR = GetChildAt(C, Idx, TC);
  if (!AR)
    return AR;

  StringRef ActualText = TC->getText();
  if (ActualText != Text)
    return ::testing::AssertionFailure()
        << "TextComment has text \"" << ActualText.str() << "\", "
           "expected \"" << Text.str() << "\"";

  if (!TC->hasTrailingNewline())
    return ::testing::AssertionFailure()
        << "TextComment has no trailing newline";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasBlockCommandAt(const Comment *C,
                                             const CommandTraits &Traits,
                                             size_t Idx,
                                             BlockCommandComment *&BCC,
                                             StringRef Name,
                                             ParagraphComment *&Paragraph) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, BCC);
  if (!AR)
    return AR;

  StringRef ActualName = BCC->getCommandName(Traits);
  if (ActualName != Name)
    return ::testing::AssertionFailure()
        << "BlockCommandComment has name \"" << ActualName.str() << "\", "
           "expected \"" << Name.str() << "\"";

  Paragraph = BCC->getParagraph();

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasParamCommandAt(
                              const Comment *C,
                              const CommandTraits &Traits,
                              size_t Idx,
                              ParamCommandComment *&PCC,
                              StringRef CommandName,
                              ParamCommandComment::PassDirection Direction,
                              bool IsDirectionExplicit,
                              StringRef ParamName,
                              ParagraphComment *&Paragraph) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, PCC);
  if (!AR)
    return AR;

  StringRef ActualCommandName = PCC->getCommandName(Traits);
  if (ActualCommandName != CommandName)
    return ::testing::AssertionFailure()
        << "ParamCommandComment has name \"" << ActualCommandName.str() << "\", "
           "expected \"" << CommandName.str() << "\"";

  if (PCC->getDirection() != Direction)
    return ::testing::AssertionFailure()
        << "ParamCommandComment has direction " << PCC->getDirection() << ", "
           "expected " << Direction;

  if (PCC->isDirectionExplicit() != IsDirectionExplicit)
    return ::testing::AssertionFailure()
        << "ParamCommandComment has "
        << (PCC->isDirectionExplicit() ? "explicit" : "implicit")
        << " direction, "
           "expected " << (IsDirectionExplicit ? "explicit" : "implicit");

  if (!ParamName.empty() && !PCC->hasParamName())
    return ::testing::AssertionFailure()
        << "ParamCommandComment has no parameter name";

  StringRef ActualParamName = PCC->hasParamName() ? PCC->getParamNameAsWritten() : "";
  if (ActualParamName != ParamName)
    return ::testing::AssertionFailure()
        << "ParamCommandComment has parameter name \"" << ActualParamName.str()
        << "\", "
           "expected \"" << ParamName.str() << "\"";

  Paragraph = PCC->getParagraph();

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasTParamCommandAt(
                              const Comment *C,
                              const CommandTraits &Traits,
                              size_t Idx,
                              TParamCommandComment *&TPCC,
                              StringRef CommandName,
                              StringRef ParamName,
                              ParagraphComment *&Paragraph) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, TPCC);
  if (!AR)
    return AR;

  StringRef ActualCommandName = TPCC->getCommandName(Traits);
  if (ActualCommandName != CommandName)
    return ::testing::AssertionFailure()
        << "TParamCommandComment has name \"" << ActualCommandName.str() << "\", "
           "expected \"" << CommandName.str() << "\"";

  if (!ParamName.empty() && !TPCC->hasParamName())
    return ::testing::AssertionFailure()
        << "TParamCommandComment has no parameter name";

  StringRef ActualParamName = TPCC->hasParamName() ? TPCC->getParamNameAsWritten() : "";
  if (ActualParamName != ParamName)
    return ::testing::AssertionFailure()
        << "TParamCommandComment has parameter name \"" << ActualParamName.str()
        << "\", "
           "expected \"" << ParamName.str() << "\"";

  Paragraph = TPCC->getParagraph();

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasInlineCommandAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              InlineCommandComment *&ICC,
                                              StringRef Name) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, ICC);
  if (!AR)
    return AR;

  StringRef ActualName = ICC->getCommandName(Traits);
  if (ActualName != Name)
    return ::testing::AssertionFailure()
        << "InlineCommandComment has name \"" << ActualName.str() << "\", "
           "expected \"" << Name.str() << "\"";

  return ::testing::AssertionSuccess();
}

struct NoArgs {};

::testing::AssertionResult HasInlineCommandAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              InlineCommandComment *&ICC,
                                              StringRef Name,
                                              NoArgs) {
  ::testing::AssertionResult AR = HasInlineCommandAt(C, Traits, Idx, ICC, Name);
  if (!AR)
    return AR;

  if (ICC->getNumArgs() != 0)
    return ::testing::AssertionFailure()
        << "InlineCommandComment has " << ICC->getNumArgs() << " arg(s), "
           "expected 0";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasInlineCommandAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              InlineCommandComment *&ICC,
                                              StringRef Name,
                                              StringRef Arg) {
  ::testing::AssertionResult AR = HasInlineCommandAt(C, Traits, Idx, ICC, Name);
  if (!AR)
    return AR;

  if (ICC->getNumArgs() != 1)
    return ::testing::AssertionFailure()
        << "InlineCommandComment has " << ICC->getNumArgs() << " arg(s), "
           "expected 1";

  StringRef ActualArg = ICC->getArgText(0);
  if (ActualArg != Arg)
    return ::testing::AssertionFailure()
        << "InlineCommandComment has argument \"" << ActualArg.str() << "\", "
           "expected \"" << Arg.str() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasHTMLStartTagAt(const Comment *C,
                                             size_t Idx,
                                             HTMLStartTagComment *&HST,
                                             StringRef TagName) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, HST);
  if (!AR)
    return AR;

  StringRef ActualTagName = HST->getTagName();
  if (ActualTagName != TagName)
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment has name \"" << ActualTagName.str() << "\", "
           "expected \"" << TagName.str() << "\"";

  return ::testing::AssertionSuccess();
}

struct SelfClosing {};

::testing::AssertionResult HasHTMLStartTagAt(const Comment *C,
                                             size_t Idx,
                                             HTMLStartTagComment *&HST,
                                             StringRef TagName,
                                             SelfClosing) {
  ::testing::AssertionResult AR = HasHTMLStartTagAt(C, Idx, HST, TagName);
  if (!AR)
    return AR;

  if (!HST->isSelfClosing())
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment is not self-closing";

  return ::testing::AssertionSuccess();
}


struct NoAttrs {};

::testing::AssertionResult HasHTMLStartTagAt(const Comment *C,
                                             size_t Idx,
                                             HTMLStartTagComment *&HST,
                                             StringRef TagName,
                                             NoAttrs) {
  ::testing::AssertionResult AR = HasHTMLStartTagAt(C, Idx, HST, TagName);
  if (!AR)
    return AR;

  if (HST->isSelfClosing())
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment is self-closing";

  if (HST->getNumAttrs() != 0)
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment has " << HST->getNumAttrs() << " attr(s), "
           "expected 0";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasHTMLStartTagAt(const Comment *C,
                                             size_t Idx,
                                             HTMLStartTagComment *&HST,
                                             StringRef TagName,
                                             StringRef AttrName,
                                             StringRef AttrValue) {
  ::testing::AssertionResult AR = HasHTMLStartTagAt(C, Idx, HST, TagName);
  if (!AR)
    return AR;

  if (HST->isSelfClosing())
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment is self-closing";

  if (HST->getNumAttrs() != 1)
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment has " << HST->getNumAttrs() << " attr(s), "
           "expected 1";

  StringRef ActualName = HST->getAttr(0).Name;
  if (ActualName != AttrName)
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment has attr \"" << ActualName.str() << "\", "
           "expected \"" << AttrName.str() << "\"";

  StringRef ActualValue = HST->getAttr(0).Value;
  if (ActualValue != AttrValue)
    return ::testing::AssertionFailure()
        << "HTMLStartTagComment has attr value \"" << ActualValue.str() << "\", "
           "expected \"" << AttrValue.str() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasHTMLEndTagAt(const Comment *C,
                                           size_t Idx,
                                           HTMLEndTagComment *&HET,
                                           StringRef TagName) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, HET);
  if (!AR)
    return AR;

  StringRef ActualTagName = HET->getTagName();
  if (ActualTagName != TagName)
    return ::testing::AssertionFailure()
        << "HTMLEndTagComment has name \"" << ActualTagName.str() << "\", "
           "expected \"" << TagName.str() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasParagraphCommentAt(const Comment *C,
                                                 size_t Idx,
                                                 StringRef Text) {
  ParagraphComment *PC;

  {
    ::testing::AssertionResult AR = GetChildAt(C, Idx, PC);
    if (!AR)
      return AR;
  }

  {
    ::testing::AssertionResult AR = HasChildCount(PC, 1);
    if (!AR)
      return AR;
  }

  {
    ::testing::AssertionResult AR = HasTextAt(PC, 0, Text);
    if (!AR)
      return AR;
  }

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasVerbatimBlockAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              VerbatimBlockComment *&VBC,
                                              StringRef Name,
                                              StringRef CloseName) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, VBC);
  if (!AR)
    return AR;

  StringRef ActualName = VBC->getCommandName(Traits);
  if (ActualName != Name)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has name \"" << ActualName.str() << "\", "
           "expected \"" << Name.str() << "\"";

  StringRef ActualCloseName = VBC->getCloseName();
  if (ActualCloseName != CloseName)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has closing command name \""
        << ActualCloseName.str() << "\", "
           "expected \"" << CloseName.str() << "\"";

  return ::testing::AssertionSuccess();
}

struct NoLines {};
struct Lines {};

::testing::AssertionResult HasVerbatimBlockAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              VerbatimBlockComment *&VBC,
                                              StringRef Name,
                                              StringRef CloseName,
                                              NoLines) {
  ::testing::AssertionResult AR = HasVerbatimBlockAt(C, Traits, Idx, VBC, Name,
                                                     CloseName);
  if (!AR)
    return AR;

  if (VBC->getNumLines() != 0)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has " << VBC->getNumLines() << " lines(s), "
           "expected 0";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasVerbatimBlockAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              VerbatimBlockComment *&VBC,
                                              StringRef Name,
                                              StringRef CloseName,
                                              Lines,
                                              StringRef Line0) {
  ::testing::AssertionResult AR = HasVerbatimBlockAt(C, Traits, Idx, VBC, Name,
                                                     CloseName);
  if (!AR)
    return AR;

  if (VBC->getNumLines() != 1)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has " << VBC->getNumLines() << " lines(s), "
           "expected 1";

  StringRef ActualLine0 = VBC->getText(0);
  if (ActualLine0 != Line0)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has lines[0] \"" << ActualLine0.str() << "\", "
           "expected \"" << Line0.str() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasVerbatimBlockAt(const Comment *C,
                                              const CommandTraits &Traits,
                                              size_t Idx,
                                              VerbatimBlockComment *&VBC,
                                              StringRef Name,
                                              StringRef CloseName,
                                              Lines,
                                              StringRef Line0,
                                              StringRef Line1) {
  ::testing::AssertionResult AR = HasVerbatimBlockAt(C, Traits, Idx, VBC, Name,
                                                     CloseName);
  if (!AR)
    return AR;

  if (VBC->getNumLines() != 2)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has " << VBC->getNumLines() << " lines(s), "
           "expected 2";

  StringRef ActualLine0 = VBC->getText(0);
  if (ActualLine0 != Line0)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has lines[0] \"" << ActualLine0.str() << "\", "
           "expected \"" << Line0.str() << "\"";

  StringRef ActualLine1 = VBC->getText(1);
  if (ActualLine1 != Line1)
    return ::testing::AssertionFailure()
        << "VerbatimBlockComment has lines[1] \"" << ActualLine1.str() << "\", "
           "expected \"" << Line1.str() << "\"";

  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasVerbatimLineAt(const Comment *C,
                                             const CommandTraits &Traits,
                                             size_t Idx,
                                             VerbatimLineComment *&VLC,
                                             StringRef Name,
                                             StringRef Text) {
  ::testing::AssertionResult AR = GetChildAt(C, Idx, VLC);
  if (!AR)
    return AR;

  StringRef ActualName = VLC->getCommandName(Traits);
  if (ActualName != Name)
    return ::testing::AssertionFailure()
        << "VerbatimLineComment has name \"" << ActualName.str() << "\", "
           "expected \"" << Name.str() << "\"";

  StringRef ActualText = VLC->getText();
  if (ActualText != Text)
    return ::testing::AssertionFailure()
        << "VerbatimLineComment has text \"" << ActualText.str() << "\", "
           "expected \"" << Text.str() << "\"";

  return ::testing::AssertionSuccess();
}


TEST_F(CommentParserTest, Basic1) {
  const char *Source = "//";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 0));
}

TEST_F(CommentParserTest, Basic2) {
  const char *Source = "// Meow";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " Meow"));
}

TEST_F(CommentParserTest, Basic3) {
  const char *Source =
    "// Aaa\n"
    "// Bbb";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 2));
      ASSERT_TRUE(HasTextWithNewlineAt(PC, 0, " Aaa"));
      ASSERT_TRUE(HasTextAt(PC, 1, " Bbb"));
  }
}

TEST_F(CommentParserTest, ParagraphSplitting1) {
  const char *Sources[] = {
    ("// Aaa\n"
    "//\n"
    "// Bbb"),

    ("// Aaa\n"
    "// \n"
    "// Bbb"),

    ("// Aaa\n"
    "//\t\n"
    "// Bbb"),

    ("// Aaa\n"
    "//\n"
    "//\n"
    "// Bbb"),

    ("/**\n"
    " Aaa\n"
    "\n"
    " Bbb\n"
    "*/"),

    ("/**\n"
    " Aaa\n"
    " \n"
    " Bbb\n"
    "*/"),

    ("/**\n"
    " Aaa\n"
    "\t \n"
    " Bbb\n"
    "*/"),
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " Aaa"));
    ASSERT_TRUE(HasParagraphCommentAt(FC, 1, " Bbb"));
  }
}

TEST_F(CommentParserTest, Paragraph1) {
  const char *Source =
    "// \\brief Aaa\n"
    "//\n"
    "// Bbb";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 3));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 1, BCC, "brief", PC));

    ASSERT_TRUE(HasParagraphCommentAt(BCC, 0, " Aaa"));
  }
  ASSERT_TRUE(HasParagraphCommentAt(FC, 2, " Bbb"));
}

TEST_F(CommentParserTest, Paragraph2) {
  const char *Source = "// \\brief \\author";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 3));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 1, BCC, "brief", PC));

    ASSERT_TRUE(HasParagraphCommentAt(BCC, 0, " "));
  }
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 2, BCC, "author", PC));

    ASSERT_TRUE(GetChildAt(BCC, 0, PC));
      ASSERT_TRUE(HasChildCount(PC, 0));
  }
}

TEST_F(CommentParserTest, Paragraph3) {
  const char *Source =
    "// \\brief Aaa\n"
    "// Bbb \\author\n"
    "// Ccc";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 3));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 1, BCC, "brief", PC));

    ASSERT_TRUE(GetChildAt(BCC, 0, PC));
      ASSERT_TRUE(HasChildCount(PC, 2));
      ASSERT_TRUE(HasTextWithNewlineAt(PC, 0, " Aaa"));
      ASSERT_TRUE(HasTextAt(PC, 1, " Bbb "));
  }
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 2, BCC, "author", PC));

    ASSERT_TRUE(HasParagraphCommentAt(BCC, 0, " Ccc"));
  }
}

TEST_F(CommentParserTest, ParamCommand1) {
  const char *Source = "// \\param aaa";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 2));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    ParamCommandComment *PCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                  ParamCommandComment::In,
                                  /* IsDirectionExplicit = */ false,
                                  "aaa", PC));
    ASSERT_TRUE(HasChildCount(PCC, 1));
    ASSERT_TRUE(HasChildCount(PC, 0));
  }
}

TEST_F(CommentParserTest, ParamCommand2) {
  const char *Source = "// \\param\\brief";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 3));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    ParamCommandComment *PCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                  ParamCommandComment::In,
                                  /* IsDirectionExplicit = */ false,
                                  "", PC));
    ASSERT_TRUE(HasChildCount(PCC, 1));
    ASSERT_TRUE(HasChildCount(PC, 0));
  }
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 2, BCC, "brief", PC));
    ASSERT_TRUE(HasChildCount(PC, 0));
  }
}

TEST_F(CommentParserTest, ParamCommand3) {
  const char *Sources[] = {
    "// \\param aaa Bbb\n",
    ("// \\param\n"
    "//     aaa Bbb\n"),
    ("// \\param \n"
    "//     aaa Bbb\n"),
    ("// \\param aaa\n"
    "// Bbb\n")
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      ParamCommandComment *PCC;
      ParagraphComment *PC;
      ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                    ParamCommandComment::In,
                                    /* IsDirectionExplicit = */ false,
                                    "aaa", PC));
      ASSERT_TRUE(HasChildCount(PCC, 1));
      ASSERT_TRUE(HasParagraphCommentAt(PCC, 0, " Bbb"));
    }
  }
}

TEST_F(CommentParserTest, ParamCommand4) {
  const char *Sources[] = {
    "// \\param [in] aaa Bbb\n",
    "// \\param[in] aaa Bbb\n",
    ("// \\param\n"
    "//     [in] aaa Bbb\n"),
    ("// \\param [in]\n"
    "//     aaa Bbb\n"),
    ("// \\param [in] aaa\n"
    "// Bbb\n"),
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      ParamCommandComment *PCC;
      ParagraphComment *PC;
      ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                    ParamCommandComment::In,
                                    /* IsDirectionExplicit = */ true,
                                    "aaa", PC));
      ASSERT_TRUE(HasChildCount(PCC, 1));
      ASSERT_TRUE(HasParagraphCommentAt(PCC, 0, " Bbb"));
    }
  }
}

TEST_F(CommentParserTest, ParamCommand5) {
  const char *Sources[] = {
    "// \\param [out] aaa Bbb\n",
    "// \\param[out] aaa Bbb\n",
    ("// \\param\n"
    "//     [out] aaa Bbb\n"),
    ("// \\param [out]\n"
    "//     aaa Bbb\n"),
    ("// \\param [out] aaa\n"
    "// Bbb\n"),
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      ParamCommandComment *PCC;
      ParagraphComment *PC;
      ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                    ParamCommandComment::Out,
                                    /* IsDirectionExplicit = */ true,
                                    "aaa", PC));
      ASSERT_TRUE(HasChildCount(PCC, 1));
      ASSERT_TRUE(HasParagraphCommentAt(PCC, 0, " Bbb"));
    }
  }
}

TEST_F(CommentParserTest, ParamCommand6) {
  const char *Sources[] = {
    "// \\param [in,out] aaa Bbb\n",
    "// \\param[in,out] aaa Bbb\n",
    "// \\param [in, out] aaa Bbb\n",
    "// \\param [in,\n"
    "//     out] aaa Bbb\n",
    "// \\param [in,out]\n"
    "//     aaa Bbb\n",
    "// \\param [in,out] aaa\n"
    "// Bbb\n"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      ParamCommandComment *PCC;
      ParagraphComment *PC;
      ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                    ParamCommandComment::InOut,
                                    /* IsDirectionExplicit = */ true,
                                    "aaa", PC));
      ASSERT_TRUE(HasChildCount(PCC, 1));
      ASSERT_TRUE(HasParagraphCommentAt(PCC, 0, " Bbb"));
    }
  }
}

TEST_F(CommentParserTest, ParamCommand7) {
  const char *Source =
    "// \\param aaa \\% Bbb \\$ ccc\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 2));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    ParamCommandComment *PCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasParamCommandAt(FC, Traits, 1, PCC, "param",
                                  ParamCommandComment::In,
                                  /* IsDirectionExplicit = */ false,
                                  "aaa", PC));
    ASSERT_TRUE(HasChildCount(PCC, 1));

    ASSERT_TRUE(HasChildCount(PC, 5));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasTextAt(PC, 1, "%"));
      ASSERT_TRUE(HasTextAt(PC, 2, " Bbb "));
      ASSERT_TRUE(HasTextAt(PC, 3, "$"));
      ASSERT_TRUE(HasTextAt(PC, 4, " ccc"));
  }
}

TEST_F(CommentParserTest, TParamCommand1) {
  const char *Sources[] = {
    "// \\tparam aaa Bbb\n",
    "// \\tparam\n"
    "//     aaa Bbb\n",
    "// \\tparam \n"
    "//     aaa Bbb\n",
    "// \\tparam aaa\n"
    "// Bbb\n"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      TParamCommandComment *TPCC;
      ParagraphComment *PC;
      ASSERT_TRUE(HasTParamCommandAt(FC, Traits, 1, TPCC, "tparam",
                                     "aaa", PC));
      ASSERT_TRUE(HasChildCount(TPCC, 1));
      ASSERT_TRUE(HasParagraphCommentAt(TPCC, 0, " Bbb"));
    }
  }
}

TEST_F(CommentParserTest, TParamCommand2) {
  const char *Source = "// \\tparam\\brief";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 3));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    TParamCommandComment *TPCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasTParamCommandAt(FC, Traits, 1, TPCC, "tparam", "", PC));
    ASSERT_TRUE(HasChildCount(TPCC, 1));
    ASSERT_TRUE(HasChildCount(PC, 0));
  }
  {
    BlockCommandComment *BCC;
    ParagraphComment *PC;
    ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 2, BCC, "brief", PC));
    ASSERT_TRUE(HasChildCount(PC, 0));
  }
}


TEST_F(CommentParserTest, InlineCommand1) {
  const char *Source = "// \\c";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    InlineCommandComment *ICC;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 2));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasInlineCommandAt(PC, Traits, 1, ICC, "c", NoArgs()));
  }
}

TEST_F(CommentParserTest, InlineCommand2) {
  const char *Source = "// \\c ";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    InlineCommandComment *ICC;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 3));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasInlineCommandAt(PC, Traits, 1, ICC, "c", NoArgs()));
      ASSERT_TRUE(HasTextAt(PC, 2, " "));
  }
}

TEST_F(CommentParserTest, InlineCommand3) {
  const char *Source = "// \\c aaa\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    InlineCommandComment *ICC;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 2));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasInlineCommandAt(PC, Traits, 1, ICC, "c", "aaa"));
  }
}

TEST_F(CommentParserTest, InlineCommand4) {
  const char *Source = "// \\c aaa bbb";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    InlineCommandComment *ICC;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 3));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasInlineCommandAt(PC, Traits, 1, ICC, "c", "aaa"));
      ASSERT_TRUE(HasTextAt(PC, 2, " bbb"));
  }
}

TEST_F(CommentParserTest, InlineCommand5) {
  const char *Source = "// \\unknown aaa\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    InlineCommandComment *ICC;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 3));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasInlineCommandAt(PC, Traits, 1, ICC, "unknown", NoArgs()));
      ASSERT_TRUE(HasTextAt(PC, 2, " aaa"));
  }
}

TEST_F(CommentParserTest, HTML1) {
  const char *Sources[] = {
    "// <a",
    "// <a>",
    "// <a >"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 1));

    {
      ParagraphComment *PC;
      HTMLStartTagComment *HST;
      ASSERT_TRUE(GetChildAt(FC, 0, PC));

      ASSERT_TRUE(HasChildCount(PC, 2));
        ASSERT_TRUE(HasTextAt(PC, 0, " "));
        ASSERT_TRUE(HasHTMLStartTagAt(PC, 1, HST, "a", NoAttrs()));
    }
  }
}

TEST_F(CommentParserTest, HTML2) {
  const char *Sources[] = {
    "// <br/>",
    "// <br />"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 1));

    {
      ParagraphComment *PC;
      HTMLStartTagComment *HST;
      ASSERT_TRUE(GetChildAt(FC, 0, PC));

      ASSERT_TRUE(HasChildCount(PC, 2));
        ASSERT_TRUE(HasTextAt(PC, 0, " "));
        ASSERT_TRUE(HasHTMLStartTagAt(PC, 1, HST, "br", SelfClosing()));
    }
  }
}

TEST_F(CommentParserTest, HTML3) {
  const char *Sources[] = {
    "// <a href",
    "// <a href ",
    "// <a href>",
    "// <a href >",
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 1));

    {
      ParagraphComment *PC;
      HTMLStartTagComment *HST;
      ASSERT_TRUE(GetChildAt(FC, 0, PC));

      ASSERT_TRUE(HasChildCount(PC, 2));
        ASSERT_TRUE(HasTextAt(PC, 0, " "));
        ASSERT_TRUE(HasHTMLStartTagAt(PC, 1, HST, "a", "href", ""));
    }
  }
}

TEST_F(CommentParserTest, HTML4) {
  const char *Sources[] = {
    "// <a href=\"bbb\"",
    "// <a href=\"bbb\">",
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 1));

    {
      ParagraphComment *PC;
      HTMLStartTagComment *HST;
      ASSERT_TRUE(GetChildAt(FC, 0, PC));

      ASSERT_TRUE(HasChildCount(PC, 2));
        ASSERT_TRUE(HasTextAt(PC, 0, " "));
        ASSERT_TRUE(HasHTMLStartTagAt(PC, 1, HST, "a", "href", "bbb"));
    }
  }
}

TEST_F(CommentParserTest, HTML5) {
  const char *Sources[] = {
    "// </a",
    "// </a>",
    "// </a >"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 1));

    {
      ParagraphComment *PC;
      HTMLEndTagComment *HET;
      ASSERT_TRUE(GetChildAt(FC, 0, PC));

      ASSERT_TRUE(HasChildCount(PC, 2));
        ASSERT_TRUE(HasTextAt(PC, 0, " "));
        ASSERT_TRUE(HasHTMLEndTagAt(PC, 1, HET, "a"));
    }
  }
}

TEST_F(CommentParserTest, HTML6) {
  const char *Source =
    "// <pre>\n"
    "// Aaa\n"
    "// Bbb\n"
    "// </pre>\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    ParagraphComment *PC;
    HTMLStartTagComment *HST;
    HTMLEndTagComment *HET;
    ASSERT_TRUE(GetChildAt(FC, 0, PC));

    ASSERT_TRUE(HasChildCount(PC, 6));
      ASSERT_TRUE(HasTextAt(PC, 0, " "));
      ASSERT_TRUE(HasHTMLStartTagAt(PC, 1, HST, "pre", NoAttrs()));
      ASSERT_TRUE(HasTextWithNewlineAt(PC, 2, " Aaa"));
      ASSERT_TRUE(HasTextWithNewlineAt(PC, 3, " Bbb"));
      ASSERT_TRUE(HasTextAt(PC, 4, " "));
      ASSERT_TRUE(HasHTMLEndTagAt(PC, 5, HET, "pre"));
  }
}

TEST_F(CommentParserTest, VerbatimBlock1) {
  const char *Source = "// \\verbatim\\endverbatim\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 2));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    VerbatimBlockComment *VCC;
    ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 1, VCC,
                                   "verbatim", "endverbatim",
                                   NoLines()));
  }
}

TEST_F(CommentParserTest, VerbatimBlock2) {
  const char *Source = "// \\verbatim Aaa \\endverbatim\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 2));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    VerbatimBlockComment *VBC;
    ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 1, VBC,
                                   "verbatim", "endverbatim",
                                   Lines(), " Aaa "));
  }
}

TEST_F(CommentParserTest, VerbatimBlock3) {
  const char *Source = "// \\verbatim Aaa\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 2));

  ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
  {
    VerbatimBlockComment *VBC;
    ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 1, VBC, "verbatim", "",
                                   Lines(), " Aaa"));
  }
}

TEST_F(CommentParserTest, VerbatimBlock4) {
  const char *Source =
    "//\\verbatim\n"
    "//\\endverbatim\n";

  FullComment *FC = parseString(Source);
  ASSERT_TRUE(HasChildCount(FC, 1));

  {
    VerbatimBlockComment *VBC;
    ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 0, VBC,
                                   "verbatim", "endverbatim",
                                   NoLines()));
  }
}

TEST_F(CommentParserTest, VerbatimBlock5) {
  const char *Sources[] = {
    "//\\verbatim\n"
    "// Aaa\n"
    "//\\endverbatim\n",

    "/*\\verbatim\n"
    " * Aaa\n"
    " *\\endverbatim*/"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 1));

    {
      VerbatimBlockComment *VBC;
      ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 0, VBC,
                                     "verbatim", "endverbatim",
                                     Lines(), " Aaa"));
    }
  }
}

TEST_F(CommentParserTest, VerbatimBlock6) {
  const char *Sources[] = {
    "// \\verbatim\n"
    "// Aaa\n"
    "// \\endverbatim\n",

    "/* \\verbatim\n"
    " * Aaa\n"
    " * \\endverbatim*/"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      VerbatimBlockComment *VBC;
      ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 1, VBC,
                                     "verbatim", "endverbatim",
                                     Lines(), " Aaa"));
    }
  }
}

TEST_F(CommentParserTest, VerbatimBlock7) {
  const char *Sources[] = {
    "// \\verbatim\n"
    "// Aaa\n"
    "// Bbb\n"
    "// \\endverbatim\n",

    "/* \\verbatim\n"
    " * Aaa\n"
    " * Bbb\n"
    " * \\endverbatim*/"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      VerbatimBlockComment *VBC;
      ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 1, VBC,
                                     "verbatim", "endverbatim",
                                     Lines(), " Aaa", " Bbb"));
    }
  }
}

TEST_F(CommentParserTest, VerbatimBlock8) {
  const char *Sources[] = {
    "// \\verbatim\n"
    "// Aaa\n"
    "//\n"
    "// Bbb\n"
    "// \\endverbatim\n",

    "/* \\verbatim\n"
    " * Aaa\n"
    " *\n"
    " * Bbb\n"
    " * \\endverbatim*/"
  };
  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      VerbatimBlockComment *VBC;
      ASSERT_TRUE(HasVerbatimBlockAt(FC, Traits, 1, VBC,
                                     "verbatim", "endverbatim"));
      ASSERT_EQ(3U, VBC->getNumLines());
      ASSERT_EQ(" Aaa", VBC->getText(0));
      ASSERT_EQ("",     VBC->getText(1));
      ASSERT_EQ(" Bbb", VBC->getText(2));
    }
  }
}

TEST_F(CommentParserTest, VerbatimLine1) {
  const char *Sources[] = {
    "// \\fn",
    "// \\fn\n"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      VerbatimLineComment *VLC;
      ASSERT_TRUE(HasVerbatimLineAt(FC, Traits, 1, VLC, "fn", ""));
    }
  }
}

TEST_F(CommentParserTest, VerbatimLine2) {
  const char *Sources[] = {
    "/// \\fn void *foo(const char *zzz = \"\\$\");\n//",
    "/** \\fn void *foo(const char *zzz = \"\\$\");*/"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      VerbatimLineComment *VLC;
      ASSERT_TRUE(HasVerbatimLineAt(FC, Traits, 1, VLC, "fn",
                  " void *foo(const char *zzz = \"\\$\");"));
    }
  }
}

TEST_F(CommentParserTest, Deprecated) {
  const char *Sources[] = {
    "/** @deprecated*/",
    "/// @deprecated\n"
  };

  for (size_t i = 0, e = std::size(Sources); i != e; i++) {
    FullComment *FC = parseString(Sources[i]);
    ASSERT_TRUE(HasChildCount(FC, 2));

    ASSERT_TRUE(HasParagraphCommentAt(FC, 0, " "));
    {
      BlockCommandComment *BCC;
      ParagraphComment *PC;
      ASSERT_TRUE(HasBlockCommandAt(FC, Traits, 1, BCC, "deprecated", PC));
      ASSERT_TRUE(HasChildCount(PC, 0));
    }
  }
}

} // unnamed namespace

} // end namespace comments
} // end namespace clang

