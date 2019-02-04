//===- unittests/AST/CommentLexer.cpp ------ Comment lexer tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentLexer.h"
#include "clang/AST/CommentCommandTraits.h"
#include "clang/Basic/CommentOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace clang;

namespace clang {
namespace comments {

namespace {
class CommentLexerTest : public ::testing::Test {
protected:
  CommentLexerTest()
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

  void lexString(const char *Source, std::vector<Token> &Toks);

  StringRef getCommandName(const Token &Tok) {
    return Traits.getCommandInfo(Tok.getCommandID())->Name;
  }

  StringRef getVerbatimBlockName(const Token &Tok) {
    return Traits.getCommandInfo(Tok.getVerbatimBlockID())->Name;
  }

  StringRef getVerbatimLineName(const Token &Tok) {
    return Traits.getCommandInfo(Tok.getVerbatimLineID())->Name;
  }
};

void CommentLexerTest::lexString(const char *Source,
                                 std::vector<Token> &Toks) {
  std::unique_ptr<MemoryBuffer> Buf = MemoryBuffer::getMemBuffer(Source);
  FileID File = SourceMgr.createFileID(std::move(Buf));
  SourceLocation Begin = SourceMgr.getLocForStartOfFile(File);

  Lexer L(Allocator, Diags, Traits, Begin, Source, Source + strlen(Source));

  while (1) {
    Token Tok;
    L.lex(Tok);
    if (Tok.is(tok::eof))
      break;
    Toks.push_back(Tok);
  }
}

} // unnamed namespace

// Empty source range should be handled.
TEST_F(CommentLexerTest, Basic1) {
  const char *Source = "";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(0U, Toks.size());
}

// Empty comments should be handled.
TEST_F(CommentLexerTest, Basic2) {
  const char *Sources[] = {
    "//", "///", "//!", "///<", "//!<"
  };
  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(1U, Toks.size());

    ASSERT_EQ(tok::newline, Toks[0].getKind());
  }
}

// Empty comments should be handled.
TEST_F(CommentLexerTest, Basic3) {
  const char *Sources[] = {
    "/**/", "/***/", "/*!*/", "/**<*/", "/*!<*/"
  };
  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(2U, Toks.size());

    ASSERT_EQ(tok::newline, Toks[0].getKind());
    ASSERT_EQ(tok::newline, Toks[1].getKind());
  }
}

// Single comment with plain text.
TEST_F(CommentLexerTest, Basic4) {
  const char *Sources[] = {
    "// Meow",   "/// Meow",    "//! Meow",
    "// Meow\n", "// Meow\r\n", "//! Meow\r",
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(2U, Toks.size());

    ASSERT_EQ(tok::text,          Toks[0].getKind());
    ASSERT_EQ(StringRef(" Meow"), Toks[0].getText());

    ASSERT_EQ(tok::newline,       Toks[1].getKind());
  }
}

// Single comment with plain text.
TEST_F(CommentLexerTest, Basic5) {
  const char *Sources[] = {
    "/* Meow*/", "/** Meow*/",  "/*! Meow*/"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(3U, Toks.size());

    ASSERT_EQ(tok::text,          Toks[0].getKind());
    ASSERT_EQ(StringRef(" Meow"), Toks[0].getText());

    ASSERT_EQ(tok::newline,       Toks[1].getKind());
    ASSERT_EQ(tok::newline,       Toks[2].getKind());
  }
}

// Test newline escaping.
TEST_F(CommentLexerTest, Basic6) {
  const char *Sources[] = {
    "// Aaa\\\n"   " Bbb\\ \n"   " Ccc?" "?/\n",
    "// Aaa\\\r\n" " Bbb\\ \r\n" " Ccc?" "?/\r\n",
    "// Aaa\\\r"   " Bbb\\ \r"   " Ccc?" "?/\r"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(10U, Toks.size());

    ASSERT_EQ(tok::text,         Toks[0].getKind());
    ASSERT_EQ(StringRef(" Aaa"), Toks[0].getText());
    ASSERT_EQ(tok::text,         Toks[1].getKind());
    ASSERT_EQ(StringRef("\\"),   Toks[1].getText());
    ASSERT_EQ(tok::newline,      Toks[2].getKind());

    ASSERT_EQ(tok::text,         Toks[3].getKind());
    ASSERT_EQ(StringRef(" Bbb"), Toks[3].getText());
    ASSERT_EQ(tok::text,         Toks[4].getKind());
    ASSERT_EQ(StringRef("\\"),   Toks[4].getText());
    ASSERT_EQ(tok::text,         Toks[5].getKind());
    ASSERT_EQ(StringRef(" "),    Toks[5].getText());
    ASSERT_EQ(tok::newline,      Toks[6].getKind());

    ASSERT_EQ(tok::text,         Toks[7].getKind());
    ASSERT_EQ(StringRef(" Ccc?" "?/"), Toks[7].getText());
    ASSERT_EQ(tok::newline,      Toks[8].getKind());

    ASSERT_EQ(tok::newline,      Toks[9].getKind());
  }
}

// Check that we skip C-style aligned stars correctly.
TEST_F(CommentLexerTest, Basic7) {
  const char *Source =
    "/* Aaa\n"
    " * Bbb\r\n"
    "\t* Ccc\n"
    "  ! Ddd\n"
    "  * Eee\n"
    "  ** Fff\n"
    " */";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(15U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" Aaa"), Toks[0].getText());
  ASSERT_EQ(tok::newline,      Toks[1].getKind());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef(" Bbb"), Toks[2].getText());
  ASSERT_EQ(tok::newline,      Toks[3].getKind());

  ASSERT_EQ(tok::text,         Toks[4].getKind());
  ASSERT_EQ(StringRef(" Ccc"), Toks[4].getText());
  ASSERT_EQ(tok::newline,      Toks[5].getKind());

  ASSERT_EQ(tok::text,            Toks[6].getKind());
  ASSERT_EQ(StringRef("  ! Ddd"), Toks[6].getText());
  ASSERT_EQ(tok::newline,         Toks[7].getKind());

  ASSERT_EQ(tok::text,         Toks[8].getKind());
  ASSERT_EQ(StringRef(" Eee"), Toks[8].getText());
  ASSERT_EQ(tok::newline,      Toks[9].getKind());

  ASSERT_EQ(tok::text,          Toks[10].getKind());
  ASSERT_EQ(StringRef("* Fff"), Toks[10].getText());
  ASSERT_EQ(tok::newline,       Toks[11].getKind());

  ASSERT_EQ(tok::text,         Toks[12].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[12].getText());

  ASSERT_EQ(tok::newline,      Toks[13].getKind());
  ASSERT_EQ(tok::newline,      Toks[14].getKind());
}

// A command marker followed by comment end.
TEST_F(CommentLexerTest, DoxygenCommand1) {
  const char *Sources[] = { "//@", "///@", "//!@" };
  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(2U, Toks.size());

    ASSERT_EQ(tok::text,          Toks[0].getKind());
    ASSERT_EQ(StringRef("@"),     Toks[0].getText());

    ASSERT_EQ(tok::newline,       Toks[1].getKind());
  }
}

// A command marker followed by comment end.
TEST_F(CommentLexerTest, DoxygenCommand2) {
  const char *Sources[] = { "/*@*/", "/**@*/", "/*!@*/"};
  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(3U, Toks.size());

    ASSERT_EQ(tok::text,          Toks[0].getKind());
    ASSERT_EQ(StringRef("@"),     Toks[0].getText());

    ASSERT_EQ(tok::newline,       Toks[1].getKind());
    ASSERT_EQ(tok::newline,       Toks[2].getKind());
  }
}

// A command marker followed by comment end.
TEST_F(CommentLexerTest, DoxygenCommand3) {
  const char *Sources[] = { "/*\\*/", "/**\\*/" };
  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(3U, Toks.size());

    ASSERT_EQ(tok::text,           Toks[0].getKind());
    ASSERT_EQ(StringRef("\\"),     Toks[0].getText());

    ASSERT_EQ(tok::newline,        Toks[1].getKind());
    ASSERT_EQ(tok::newline,        Toks[2].getKind());
  }
}

// Doxygen escape sequences.
TEST_F(CommentLexerTest, DoxygenCommand4) {
  const char *Sources[] = {
    "/// \\\\ \\@ \\& \\$ \\# \\< \\> \\% \\\" \\. \\::",
    "/// @\\ @@ @& @$ @# @< @> @% @\" @. @::"
  };
  const char *Text[] = {
    " ",
    "\\", " ", "@", " ", "&", " ", "$",  " ", "#", " ",
    "<",  " ", ">", " ", "%", " ", "\"", " ", ".", " ",
    "::", ""
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(array_lengthof(Text), Toks.size());

    for (size_t j = 0, e = Toks.size(); j != e; j++) {
      if(Toks[j].is(tok::text)) {
        ASSERT_EQ(StringRef(Text[j]), Toks[j].getText())
          << "index " << i;
      }
    }
  }
}

// A command marker followed by a non-letter that is not a part of an escape
// sequence.
TEST_F(CommentLexerTest, DoxygenCommand5) {
  const char *Source = "/// \\^ \\0";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(6U, Toks.size());

  ASSERT_EQ(tok::text,       Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[0].getText());

  ASSERT_EQ(tok::text,       Toks[1].getKind());
  ASSERT_EQ(StringRef("\\"), Toks[1].getText());

  ASSERT_EQ(tok::text,       Toks[2].getKind());
  ASSERT_EQ(StringRef("^ "), Toks[2].getText());

  ASSERT_EQ(tok::text,       Toks[3].getKind());
  ASSERT_EQ(StringRef("\\"), Toks[3].getText());

  ASSERT_EQ(tok::text,       Toks[4].getKind());
  ASSERT_EQ(StringRef("0"),  Toks[4].getText());

  ASSERT_EQ(tok::newline,    Toks[5].getKind());
}

TEST_F(CommentLexerTest, DoxygenCommand6) {
  const char *Source = "/// \\brief Aaa.";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::backslash_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("brief"), getCommandName(Toks[1]));

  ASSERT_EQ(tok::text,          Toks[2].getKind());
  ASSERT_EQ(StringRef(" Aaa."), Toks[2].getText());

  ASSERT_EQ(tok::newline,       Toks[3].getKind());
}

TEST_F(CommentLexerTest, DoxygenCommand7) {
  const char *Source = "/// \\em\\em \\em\t\\em\n";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(8U, Toks.size());

  ASSERT_EQ(tok::text,       Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[0].getText());

  ASSERT_EQ(tok::backslash_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[1]));

  ASSERT_EQ(tok::backslash_command, Toks[2].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[2]));

  ASSERT_EQ(tok::text,       Toks[3].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[3].getText());

  ASSERT_EQ(tok::backslash_command, Toks[4].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[4]));

  ASSERT_EQ(tok::text,       Toks[5].getKind());
  ASSERT_EQ(StringRef("\t"), Toks[5].getText());

  ASSERT_EQ(tok::backslash_command, Toks[6].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[6]));

  ASSERT_EQ(tok::newline,    Toks[7].getKind());
}

TEST_F(CommentLexerTest, DoxygenCommand8) {
  const char *Source = "/// @em@em @em\t@em\n";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(8U, Toks.size());

  ASSERT_EQ(tok::text,       Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[0].getText());

  ASSERT_EQ(tok::at_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[1]));

  ASSERT_EQ(tok::at_command, Toks[2].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[2]));

  ASSERT_EQ(tok::text,       Toks[3].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[3].getText());

  ASSERT_EQ(tok::at_command, Toks[4].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[4]));

  ASSERT_EQ(tok::text,       Toks[5].getKind());
  ASSERT_EQ(StringRef("\t"), Toks[5].getText());

  ASSERT_EQ(tok::at_command, Toks[6].getKind());
  ASSERT_EQ(StringRef("em"), getCommandName(Toks[6]));

  ASSERT_EQ(tok::newline,    Toks[7].getKind());
}

TEST_F(CommentLexerTest, DoxygenCommand9) {
  const char *Source = "/// \\aaa\\bbb \\ccc\t\\ddd\n";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(8U, Toks.size());

  ASSERT_EQ(tok::text,        Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),   Toks[0].getText());

  ASSERT_EQ(tok::unknown_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("aaa"), Toks[1].getUnknownCommandName());

  ASSERT_EQ(tok::unknown_command, Toks[2].getKind());
  ASSERT_EQ(StringRef("bbb"), Toks[2].getUnknownCommandName());

  ASSERT_EQ(tok::text,        Toks[3].getKind());
  ASSERT_EQ(StringRef(" "),   Toks[3].getText());

  ASSERT_EQ(tok::unknown_command, Toks[4].getKind());
  ASSERT_EQ(StringRef("ccc"), Toks[4].getUnknownCommandName());

  ASSERT_EQ(tok::text,        Toks[5].getKind());
  ASSERT_EQ(StringRef("\t"),  Toks[5].getText());

  ASSERT_EQ(tok::unknown_command, Toks[6].getKind());
  ASSERT_EQ(StringRef("ddd"), Toks[6].getUnknownCommandName());

  ASSERT_EQ(tok::newline,     Toks[7].getKind());
}

TEST_F(CommentLexerTest, DoxygenCommand10) {
  const char *Source = "// \\c\n";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,      Toks[0].getKind());
  ASSERT_EQ(StringRef(" "), Toks[0].getText());

  ASSERT_EQ(tok::backslash_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("c"), getCommandName(Toks[1]));

  ASSERT_EQ(tok::newline,   Toks[2].getKind());
}

TEST_F(CommentLexerTest, RegisterCustomBlockCommand) {
  const char *Source =
    "/// \\NewBlockCommand Aaa.\n"
    "/// @NewBlockCommand Aaa.\n";

  Traits.registerBlockCommand(StringRef("NewBlockCommand"));

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(8U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::backslash_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("NewBlockCommand"), getCommandName(Toks[1]));

  ASSERT_EQ(tok::text,          Toks[2].getKind());
  ASSERT_EQ(StringRef(" Aaa."), Toks[2].getText());

  ASSERT_EQ(tok::newline,       Toks[3].getKind());

  ASSERT_EQ(tok::text,          Toks[4].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[4].getText());

  ASSERT_EQ(tok::at_command,    Toks[5].getKind());
  ASSERT_EQ(StringRef("NewBlockCommand"), getCommandName(Toks[5]));

  ASSERT_EQ(tok::text,          Toks[6].getKind());
  ASSERT_EQ(StringRef(" Aaa."), Toks[6].getText());

  ASSERT_EQ(tok::newline,       Toks[7].getKind());
}

TEST_F(CommentLexerTest, RegisterMultipleBlockCommands) {
  const char *Source =
    "/// \\Foo\n"
    "/// \\Bar Baz\n"
    "/// \\Blech quux=corge\n";

  Traits.registerBlockCommand(StringRef("Foo"));
  Traits.registerBlockCommand(StringRef("Bar"));
  Traits.registerBlockCommand(StringRef("Blech"));

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(11U, Toks.size());

  ASSERT_EQ(tok::text,      Toks[0].getKind());
  ASSERT_EQ(StringRef(" "), Toks[0].getText());

  ASSERT_EQ(tok::backslash_command, Toks[1].getKind());
  ASSERT_EQ(StringRef("Foo"), getCommandName(Toks[1]));

  ASSERT_EQ(tok::newline,     Toks[2].getKind());

  ASSERT_EQ(tok::text,      Toks[3].getKind());
  ASSERT_EQ(StringRef(" "), Toks[3].getText());

  ASSERT_EQ(tok::backslash_command, Toks[4].getKind());
  ASSERT_EQ(StringRef("Bar"), getCommandName(Toks[4]));

  ASSERT_EQ(tok::text,         Toks[5].getKind());
  ASSERT_EQ(StringRef(" Baz"), Toks[5].getText());

  ASSERT_EQ(tok::newline,     Toks[6].getKind());

  ASSERT_EQ(tok::text,      Toks[7].getKind());
  ASSERT_EQ(StringRef(" "), Toks[7].getText());

  ASSERT_EQ(tok::backslash_command, Toks[8].getKind());
  ASSERT_EQ(StringRef("Blech"), getCommandName(Toks[8]));

  ASSERT_EQ(tok::text,                Toks[9].getKind());
  ASSERT_EQ(StringRef(" quux=corge"), Toks[9].getText());

  ASSERT_EQ(tok::newline,     Toks[10].getKind());
}

// Empty verbatim block.
TEST_F(CommentLexerTest, VerbatimBlock1) {
  const char *Sources[] = {
    "/// \\verbatim\\endverbatim\n//",
    "/** \\verbatim\\endverbatim*/"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(5U, Toks.size());

    ASSERT_EQ(tok::text,                 Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),            Toks[0].getText());

    ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
    ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

    ASSERT_EQ(tok::verbatim_block_end,   Toks[2].getKind());
    ASSERT_EQ(StringRef("endverbatim"),  getVerbatimBlockName(Toks[2]));

    ASSERT_EQ(tok::newline,              Toks[3].getKind());
    ASSERT_EQ(tok::newline,              Toks[4].getKind());
  }
}

// Empty verbatim block without an end command.
TEST_F(CommentLexerTest, VerbatimBlock2) {
  const char *Source = "/// \\verbatim";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,                 Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[0].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
  ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

  ASSERT_EQ(tok::newline,              Toks[2].getKind());
}

// Empty verbatim block without an end command.
TEST_F(CommentLexerTest, VerbatimBlock3) {
  const char *Source = "/** \\verbatim*/";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,                 Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[0].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
  ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

  ASSERT_EQ(tok::newline,              Toks[2].getKind());
  ASSERT_EQ(tok::newline,              Toks[3].getKind());
}

// Single-line verbatim block.
TEST_F(CommentLexerTest, VerbatimBlock4) {
  const char *Sources[] = {
    "/// Meow \\verbatim aaa \\endverbatim\n//",
    "/** Meow \\verbatim aaa \\endverbatim*/"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(6U, Toks.size());

    ASSERT_EQ(tok::text,                 Toks[0].getKind());
    ASSERT_EQ(StringRef(" Meow "),       Toks[0].getText());

    ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
    ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

    ASSERT_EQ(tok::verbatim_block_line,  Toks[2].getKind());
    ASSERT_EQ(StringRef(" aaa "),        Toks[2].getVerbatimBlockText());

    ASSERT_EQ(tok::verbatim_block_end,   Toks[3].getKind());
    ASSERT_EQ(StringRef("endverbatim"),  getVerbatimBlockName(Toks[3]));

    ASSERT_EQ(tok::newline,              Toks[4].getKind());
    ASSERT_EQ(tok::newline,              Toks[5].getKind());
  }
}

// Single-line verbatim block without an end command.
TEST_F(CommentLexerTest, VerbatimBlock5) {
  const char *Sources[] = {
    "/// Meow \\verbatim aaa \n//",
    "/** Meow \\verbatim aaa */"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(5U, Toks.size());

    ASSERT_EQ(tok::text,                 Toks[0].getKind());
    ASSERT_EQ(StringRef(" Meow "),       Toks[0].getText());

    ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
    ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

    ASSERT_EQ(tok::verbatim_block_line,  Toks[2].getKind());
    ASSERT_EQ(StringRef(" aaa "),        Toks[2].getVerbatimBlockText());

    ASSERT_EQ(tok::newline,              Toks[3].getKind());
    ASSERT_EQ(tok::newline,              Toks[4].getKind());
  }
}

TEST_F(CommentLexerTest, VerbatimBlock6) {
  const char *Source =
    "// \\verbatim\n"
    "// Aaa\n"
    "//\n"
    "// Bbb\n"
    "// \\endverbatim\n";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(10U, Toks.size());

  ASSERT_EQ(tok::text,                 Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[0].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
  ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

  ASSERT_EQ(tok::newline,              Toks[2].getKind());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[3].getKind());
  ASSERT_EQ(StringRef(" Aaa"),         Toks[3].getVerbatimBlockText());

  ASSERT_EQ(tok::newline,              Toks[4].getKind());

  ASSERT_EQ(tok::newline,              Toks[5].getKind());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[6].getKind());
  ASSERT_EQ(StringRef(" Bbb"),         Toks[6].getVerbatimBlockText());

  ASSERT_EQ(tok::newline,              Toks[7].getKind());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[8].getKind());
  ASSERT_EQ(StringRef("endverbatim"),  getVerbatimBlockName(Toks[8]));

  ASSERT_EQ(tok::newline,              Toks[9].getKind());
}

TEST_F(CommentLexerTest, VerbatimBlock7) {
  const char *Source =
    "/* \\verbatim\n"
    " * Aaa\n"
    " *\n"
    " * Bbb\n"
    " * \\endverbatim\n"
    " */";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(10U, Toks.size());

  ASSERT_EQ(tok::text,                 Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[0].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
  ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

  ASSERT_EQ(tok::verbatim_block_line,  Toks[2].getKind());
  ASSERT_EQ(StringRef(" Aaa"),         Toks[2].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[3].getKind());
  ASSERT_EQ(StringRef(""),             Toks[3].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[4].getKind());
  ASSERT_EQ(StringRef(" Bbb"),         Toks[4].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[5].getKind());
  ASSERT_EQ(StringRef("endverbatim"),  getVerbatimBlockName(Toks[5]));

  ASSERT_EQ(tok::newline,              Toks[6].getKind());

  ASSERT_EQ(tok::text,                 Toks[7].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[7].getText());

  ASSERT_EQ(tok::newline,              Toks[8].getKind());
  ASSERT_EQ(tok::newline,              Toks[9].getKind());
}

// Complex test for verbatim blocks.
TEST_F(CommentLexerTest, VerbatimBlock8) {
  const char *Source =
    "/* Meow \\verbatim aaa\\$\\@\n"
    "bbb \\endverbati\r"
    "ccc\r\n"
    "ddd \\endverbatim Blah \\verbatim eee\n"
    "\\endverbatim BlahBlah*/";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(14U, Toks.size());

  ASSERT_EQ(tok::text,                 Toks[0].getKind());
  ASSERT_EQ(StringRef(" Meow "),       Toks[0].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
  ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[1]));

  ASSERT_EQ(tok::verbatim_block_line,  Toks[2].getKind());
  ASSERT_EQ(StringRef(" aaa\\$\\@"),   Toks[2].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[3].getKind());
  ASSERT_EQ(StringRef("bbb \\endverbati"), Toks[3].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[4].getKind());
  ASSERT_EQ(StringRef("ccc"),          Toks[4].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_line,  Toks[5].getKind());
  ASSERT_EQ(StringRef("ddd "),         Toks[5].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[6].getKind());
  ASSERT_EQ(StringRef("endverbatim"),  getVerbatimBlockName(Toks[6]));

  ASSERT_EQ(tok::text,                 Toks[7].getKind());
  ASSERT_EQ(StringRef(" Blah "),       Toks[7].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[8].getKind());
  ASSERT_EQ(StringRef("verbatim"),     getVerbatimBlockName(Toks[8]));

  ASSERT_EQ(tok::verbatim_block_line,  Toks[9].getKind());
  ASSERT_EQ(StringRef(" eee"),         Toks[9].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[10].getKind());
  ASSERT_EQ(StringRef("endverbatim"),  getVerbatimBlockName(Toks[10]));

  ASSERT_EQ(tok::text,                 Toks[11].getKind());
  ASSERT_EQ(StringRef(" BlahBlah"),    Toks[11].getText());

  ASSERT_EQ(tok::newline,              Toks[12].getKind());
  ASSERT_EQ(tok::newline,              Toks[13].getKind());
}

// LaTeX verbatim blocks.
TEST_F(CommentLexerTest, VerbatimBlock9) {
  const char *Source =
    "/// \\f$ Aaa \\f$ \\f[ Bbb \\f] \\f{ Ccc \\f}";
  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(13U, Toks.size());

  ASSERT_EQ(tok::text,                 Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[0].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[1].getKind());
  ASSERT_EQ(StringRef("f$"),           getVerbatimBlockName(Toks[1]));

  ASSERT_EQ(tok::verbatim_block_line,  Toks[2].getKind());
  ASSERT_EQ(StringRef(" Aaa "),        Toks[2].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[3].getKind());
  ASSERT_EQ(StringRef("f$"),           getVerbatimBlockName(Toks[3]));

  ASSERT_EQ(tok::text,                 Toks[4].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[4].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[5].getKind());
  ASSERT_EQ(StringRef("f["),           getVerbatimBlockName(Toks[5]));

  ASSERT_EQ(tok::verbatim_block_line,  Toks[6].getKind());
  ASSERT_EQ(StringRef(" Bbb "),        Toks[6].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[7].getKind());
  ASSERT_EQ(StringRef("f]"),           getVerbatimBlockName(Toks[7]));

  ASSERT_EQ(tok::text,                 Toks[8].getKind());
  ASSERT_EQ(StringRef(" "),            Toks[8].getText());

  ASSERT_EQ(tok::verbatim_block_begin, Toks[9].getKind());
  ASSERT_EQ(StringRef("f{"),           getVerbatimBlockName(Toks[9]));

  ASSERT_EQ(tok::verbatim_block_line,  Toks[10].getKind());
  ASSERT_EQ(StringRef(" Ccc "),        Toks[10].getVerbatimBlockText());

  ASSERT_EQ(tok::verbatim_block_end,   Toks[11].getKind());
  ASSERT_EQ(StringRef("f}"),           getVerbatimBlockName(Toks[11]));

  ASSERT_EQ(tok::newline,              Toks[12].getKind());
}

// Empty verbatim line.
TEST_F(CommentLexerTest, VerbatimLine1) {
  const char *Sources[] = {
    "/// \\fn\n//",
    "/** \\fn*/"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(4U, Toks.size());

    ASSERT_EQ(tok::text,               Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),          Toks[0].getText());

    ASSERT_EQ(tok::verbatim_line_name, Toks[1].getKind());
    ASSERT_EQ(StringRef("fn"),         getVerbatimLineName(Toks[1]));

    ASSERT_EQ(tok::newline,            Toks[2].getKind());
    ASSERT_EQ(tok::newline,            Toks[3].getKind());
  }
}

// Verbatim line with Doxygen escape sequences, which should not be expanded.
TEST_F(CommentLexerTest, VerbatimLine2) {
  const char *Sources[] = {
    "/// \\fn void *foo(const char *zzz = \"\\$\");\n//",
    "/** \\fn void *foo(const char *zzz = \"\\$\");*/"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(5U, Toks.size());

    ASSERT_EQ(tok::text,               Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),          Toks[0].getText());

    ASSERT_EQ(tok::verbatim_line_name, Toks[1].getKind());
    ASSERT_EQ(StringRef("fn"),         getVerbatimLineName(Toks[1]));

    ASSERT_EQ(tok::verbatim_line_text, Toks[2].getKind());
    ASSERT_EQ(StringRef(" void *foo(const char *zzz = \"\\$\");"),
                                       Toks[2].getVerbatimLineText());

    ASSERT_EQ(tok::newline,            Toks[3].getKind());
    ASSERT_EQ(tok::newline,            Toks[4].getKind());
  }
}

// Verbatim line should not eat anything from next source line.
TEST_F(CommentLexerTest, VerbatimLine3) {
  const char *Source =
    "/** \\fn void *foo(const char *zzz = \"\\$\");\n"
    " * Meow\n"
    " */";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(9U, Toks.size());

  ASSERT_EQ(tok::text,               Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),          Toks[0].getText());

  ASSERT_EQ(tok::verbatim_line_name, Toks[1].getKind());
  ASSERT_EQ(StringRef("fn"),         getVerbatimLineName(Toks[1]));

  ASSERT_EQ(tok::verbatim_line_text, Toks[2].getKind());
  ASSERT_EQ(StringRef(" void *foo(const char *zzz = \"\\$\");"),
                                     Toks[2].getVerbatimLineText());
  ASSERT_EQ(tok::newline,            Toks[3].getKind());

  ASSERT_EQ(tok::text,               Toks[4].getKind());
  ASSERT_EQ(StringRef(" Meow"),      Toks[4].getText());
  ASSERT_EQ(tok::newline,            Toks[5].getKind());

  ASSERT_EQ(tok::text,               Toks[6].getKind());
  ASSERT_EQ(StringRef(" "),          Toks[6].getText());

  ASSERT_EQ(tok::newline,            Toks[7].getKind());
  ASSERT_EQ(tok::newline,            Toks[8].getKind());
}

TEST_F(CommentLexerTest, HTML1) {
  const char *Source =
    "// <";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,      Toks[0].getKind());
  ASSERT_EQ(StringRef(" "), Toks[0].getText());

  ASSERT_EQ(tok::text,      Toks[1].getKind());
  ASSERT_EQ(StringRef("<"), Toks[1].getText());

  ASSERT_EQ(tok::newline,   Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTML2) {
  const char *Source =
    "// a<2";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,       Toks[0].getKind());
  ASSERT_EQ(StringRef(" a"), Toks[0].getText());

  ASSERT_EQ(tok::text,       Toks[1].getKind());
  ASSERT_EQ(StringRef("<"),  Toks[1].getText());

  ASSERT_EQ(tok::text,       Toks[2].getKind());
  ASSERT_EQ(StringRef("2"),  Toks[2].getText());

  ASSERT_EQ(tok::newline,    Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTML3) {
  const char *Source =
    "// < img";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("<"),    Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef(" img"), Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTML4) {
  const char *Sources[] = {
    "// <img",
    "// <img "
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(3U, Toks.size());

    ASSERT_EQ(tok::text,           Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),      Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::newline,        Toks[2].getKind());
  }
}

TEST_F(CommentLexerTest, HTML5) {
  const char *Source =
    "// <img 42";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,           Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),      Toks[0].getText());

  ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
  ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

  ASSERT_EQ(tok::text,           Toks[2].getKind());
  ASSERT_EQ(StringRef("42"),     Toks[2].getText());

  ASSERT_EQ(tok::newline,        Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTML6) {
  const char *Source = "// <img> Meow";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(5U, Toks.size());

  ASSERT_EQ(tok::text,           Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),      Toks[0].getText());

  ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
  ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

  ASSERT_EQ(tok::html_greater,   Toks[2].getKind());

  ASSERT_EQ(tok::text,           Toks[3].getKind());
  ASSERT_EQ(StringRef(" Meow"),  Toks[3].getText());

  ASSERT_EQ(tok::newline,        Toks[4].getKind());
}

TEST_F(CommentLexerTest, HTML7) {
  const char *Source = "// <img=";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,           Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),      Toks[0].getText());

  ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
  ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

  ASSERT_EQ(tok::text,           Toks[2].getKind());
  ASSERT_EQ(StringRef("="),      Toks[2].getText());

  ASSERT_EQ(tok::newline,        Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTML8) {
  const char *Source = "// <img src=> Meow";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(7U, Toks.size());

  ASSERT_EQ(tok::text,           Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),      Toks[0].getText());

  ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
  ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

  ASSERT_EQ(tok::html_ident,     Toks[2].getKind());
  ASSERT_EQ(StringRef("src"),   Toks[2].getHTMLIdent());

  ASSERT_EQ(tok::html_equals,    Toks[3].getKind());

  ASSERT_EQ(tok::html_greater,   Toks[4].getKind());

  ASSERT_EQ(tok::text,           Toks[5].getKind());
  ASSERT_EQ(StringRef(" Meow"),  Toks[5].getText());

  ASSERT_EQ(tok::newline,        Toks[6].getKind());
}

TEST_F(CommentLexerTest, HTML9) {
  const char *Sources[] = {
    "// <img src",
    "// <img src "
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(4U, Toks.size());

    ASSERT_EQ(tok::text,           Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),      Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::html_ident,     Toks[2].getKind());
    ASSERT_EQ(StringRef("src"),    Toks[2].getHTMLIdent());

    ASSERT_EQ(tok::newline,        Toks[3].getKind());
  }
}

TEST_F(CommentLexerTest, HTML10) {
  const char *Sources[] = {
    "// <img src=",
    "// <img src ="
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(5U, Toks.size());

    ASSERT_EQ(tok::text,           Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),      Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::html_ident,     Toks[2].getKind());
    ASSERT_EQ(StringRef("src"),    Toks[2].getHTMLIdent());

    ASSERT_EQ(tok::html_equals,    Toks[3].getKind());

    ASSERT_EQ(tok::newline,        Toks[4].getKind());
  }
}

TEST_F(CommentLexerTest, HTML11) {
  const char *Sources[] = {
    "// <img src=\"",
    "// <img src = \"",
    "// <img src=\'",
    "// <img src = \'"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(6U, Toks.size());

    ASSERT_EQ(tok::text,               Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),          Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag,     Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),        Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::html_ident,         Toks[2].getKind());
    ASSERT_EQ(StringRef("src"),        Toks[2].getHTMLIdent());

    ASSERT_EQ(tok::html_equals,        Toks[3].getKind());

    ASSERT_EQ(tok::html_quoted_string, Toks[4].getKind());
    ASSERT_EQ(StringRef(""),           Toks[4].getHTMLQuotedString());

    ASSERT_EQ(tok::newline,            Toks[5].getKind());
  }
}

TEST_F(CommentLexerTest, HTML12) {
  const char *Source = "// <img src=@";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(6U, Toks.size());

  ASSERT_EQ(tok::text,           Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),      Toks[0].getText());

  ASSERT_EQ(tok::html_start_tag, Toks[1].getKind());
  ASSERT_EQ(StringRef("img"),    Toks[1].getHTMLTagStartName());

  ASSERT_EQ(tok::html_ident,     Toks[2].getKind());
  ASSERT_EQ(StringRef("src"),    Toks[2].getHTMLIdent());

  ASSERT_EQ(tok::html_equals,    Toks[3].getKind());

  ASSERT_EQ(tok::text,           Toks[4].getKind());
  ASSERT_EQ(StringRef("@"),      Toks[4].getText());

  ASSERT_EQ(tok::newline,        Toks[5].getKind());
}

TEST_F(CommentLexerTest, HTML13) {
  const char *Sources[] = {
    "// <img src=\"val\\\"\\'val",
    "// <img src=\"val\\\"\\'val\"",
    "// <img src=\'val\\\"\\'val",
    "// <img src=\'val\\\"\\'val\'"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(6U, Toks.size());

    ASSERT_EQ(tok::text,                  Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),             Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag,        Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),           Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::html_ident,            Toks[2].getKind());
    ASSERT_EQ(StringRef("src"),           Toks[2].getHTMLIdent());

    ASSERT_EQ(tok::html_equals,           Toks[3].getKind());

    ASSERT_EQ(tok::html_quoted_string,    Toks[4].getKind());
    ASSERT_EQ(StringRef("val\\\"\\'val"), Toks[4].getHTMLQuotedString());

    ASSERT_EQ(tok::newline,               Toks[5].getKind());
  }
}

TEST_F(CommentLexerTest, HTML14) {
  const char *Sources[] = {
    "// <img src=\"val\\\"\\'val\">",
    "// <img src=\'val\\\"\\'val\'>"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(7U, Toks.size());

    ASSERT_EQ(tok::text,                  Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),             Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag,        Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),           Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::html_ident,            Toks[2].getKind());
    ASSERT_EQ(StringRef("src"),           Toks[2].getHTMLIdent());

    ASSERT_EQ(tok::html_equals,           Toks[3].getKind());

    ASSERT_EQ(tok::html_quoted_string,    Toks[4].getKind());
    ASSERT_EQ(StringRef("val\\\"\\'val"), Toks[4].getHTMLQuotedString());

    ASSERT_EQ(tok::html_greater,          Toks[5].getKind());

    ASSERT_EQ(tok::newline,               Toks[6].getKind());
  }
}

TEST_F(CommentLexerTest, HTML15) {
  const char *Sources[] = {
    "// <img/>",
    "// <img />"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(4U, Toks.size());

    ASSERT_EQ(tok::text,               Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),          Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag,     Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),        Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::html_slash_greater, Toks[2].getKind());

    ASSERT_EQ(tok::newline,            Toks[3].getKind());
  }
}

TEST_F(CommentLexerTest, HTML16) {
  const char *Sources[] = {
    "// <img/ Aaa",
    "// <img / Aaa"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(5U, Toks.size());

    ASSERT_EQ(tok::text,               Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),          Toks[0].getText());

    ASSERT_EQ(tok::html_start_tag,     Toks[1].getKind());
    ASSERT_EQ(StringRef("img"),        Toks[1].getHTMLTagStartName());

    ASSERT_EQ(tok::text,               Toks[2].getKind());
    ASSERT_EQ(StringRef("/"),          Toks[2].getText());

    ASSERT_EQ(tok::text,               Toks[3].getKind());
    ASSERT_EQ(StringRef(" Aaa"),       Toks[3].getText());

    ASSERT_EQ(tok::newline,            Toks[4].getKind());
  }
}

TEST_F(CommentLexerTest, HTML17) {
  const char *Source = "// </";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,       Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[0].getText());

  ASSERT_EQ(tok::text,       Toks[1].getKind());
  ASSERT_EQ(StringRef("</"), Toks[1].getText());

  ASSERT_EQ(tok::newline,    Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTML18) {
  const char *Source = "// </@";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,       Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),  Toks[0].getText());

  ASSERT_EQ(tok::text,       Toks[1].getKind());
  ASSERT_EQ(StringRef("</"), Toks[1].getText());

  ASSERT_EQ(tok::text,       Toks[2].getKind());
  ASSERT_EQ(StringRef("@"),  Toks[2].getText());

  ASSERT_EQ(tok::newline,    Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTML19) {
  const char *Source = "// </img";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::html_end_tag, Toks[1].getKind());
  ASSERT_EQ(StringRef("img"),  Toks[1].getHTMLTagEndName());

  ASSERT_EQ(tok::newline,      Toks[2].getKind());
}

TEST_F(CommentLexerTest, NotAKnownHTMLTag1) {
  const char *Source = "// <tag>";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("<tag"), Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef(">"),    Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, NotAKnownHTMLTag2) {
  const char *Source = "// </tag>";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::text,          Toks[1].getKind());
  ASSERT_EQ(StringRef("</tag"), Toks[1].getText());

  ASSERT_EQ(tok::text,          Toks[2].getKind());
  ASSERT_EQ(StringRef(">"),     Toks[2].getText());

  ASSERT_EQ(tok::newline,       Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences1) {
  const char *Source = "// &";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&"),    Toks[1].getText());

  ASSERT_EQ(tok::newline,      Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences2) {
  const char *Source = "// &!";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&"),    Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef("!"),    Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences3) {
  const char *Source = "// &amp";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&amp"), Toks[1].getText());

  ASSERT_EQ(tok::newline,      Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences4) {
  const char *Source = "// &amp!";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&amp"), Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef("!"),    Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences5) {
  const char *Source = "// &#";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&#"),   Toks[1].getText());

  ASSERT_EQ(tok::newline,      Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences6) {
  const char *Source = "// &#a";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&#"),   Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef("a"),    Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences7) {
  const char *Source = "// &#42";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&#42"), Toks[1].getText());

  ASSERT_EQ(tok::newline,      Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences8) {
  const char *Source = "// &#42a";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&#42"), Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef("a"),    Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences9) {
  const char *Source = "// &#x";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&#x"),  Toks[1].getText());

  ASSERT_EQ(tok::newline,      Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences10) {
  const char *Source = "// &#xz";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,         Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),    Toks[0].getText());

  ASSERT_EQ(tok::text,         Toks[1].getKind());
  ASSERT_EQ(StringRef("&#x"),  Toks[1].getText());

  ASSERT_EQ(tok::text,         Toks[2].getKind());
  ASSERT_EQ(StringRef("z"),    Toks[2].getText());

  ASSERT_EQ(tok::newline,      Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences11) {
  const char *Source = "// &#xab";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::text,          Toks[1].getKind());
  ASSERT_EQ(StringRef("&#xab"), Toks[1].getText());

  ASSERT_EQ(tok::newline,       Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences12) {
  const char *Source = "// &#xaBz";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::text,          Toks[1].getKind());
  ASSERT_EQ(StringRef("&#xaB"), Toks[1].getText());

  ASSERT_EQ(tok::text,          Toks[2].getKind());
  ASSERT_EQ(StringRef("z"),     Toks[2].getText());

  ASSERT_EQ(tok::newline,       Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences13) {
  const char *Source = "// &amp;";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(3U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::text,          Toks[1].getKind());
  ASSERT_EQ(StringRef("&"),     Toks[1].getText());

  ASSERT_EQ(tok::newline,       Toks[2].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences14) {
  const char *Source = "// &amp;&lt;";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::text,          Toks[1].getKind());
  ASSERT_EQ(StringRef("&"),     Toks[1].getText());

  ASSERT_EQ(tok::text,          Toks[2].getKind());
  ASSERT_EQ(StringRef("<"),     Toks[2].getText());

  ASSERT_EQ(tok::newline,       Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences15) {
  const char *Source = "// &amp; meow";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(4U, Toks.size());

  ASSERT_EQ(tok::text,          Toks[0].getKind());
  ASSERT_EQ(StringRef(" "),     Toks[0].getText());

  ASSERT_EQ(tok::text,          Toks[1].getKind());
  ASSERT_EQ(StringRef("&"),     Toks[1].getText());

  ASSERT_EQ(tok::text,          Toks[2].getKind());
  ASSERT_EQ(StringRef(" meow"), Toks[2].getText());

  ASSERT_EQ(tok::newline,       Toks[3].getKind());
}

TEST_F(CommentLexerTest, HTMLCharacterReferences16) {
  const char *Sources[] = {
    "// &#61;",
    "// &#x3d;",
    "// &#X3d;",
    "// &#X3D;"
  };

  for (size_t i = 0, e = array_lengthof(Sources); i != e; i++) {
    std::vector<Token> Toks;

    lexString(Sources[i], Toks);

    ASSERT_EQ(3U, Toks.size());

    ASSERT_EQ(tok::text,          Toks[0].getKind());
    ASSERT_EQ(StringRef(" "),     Toks[0].getText());

    ASSERT_EQ(tok::text,          Toks[1].getKind());
    ASSERT_EQ(StringRef("="),     Toks[1].getText());

    ASSERT_EQ(tok::newline,       Toks[2].getKind());
  }
}

TEST_F(CommentLexerTest, MultipleComments) {
  const char *Source =
    "// Aaa\n"
    "/// Bbb\n"
    "/* Ccc\n"
    " * Ddd*/\n"
    "/** Eee*/";

  std::vector<Token> Toks;

  lexString(Source, Toks);

  ASSERT_EQ(12U, Toks.size());

  ASSERT_EQ(tok::text,           Toks[0].getKind());
  ASSERT_EQ(StringRef(" Aaa"),   Toks[0].getText());
  ASSERT_EQ(tok::newline,        Toks[1].getKind());

  ASSERT_EQ(tok::text,           Toks[2].getKind());
  ASSERT_EQ(StringRef(" Bbb"),   Toks[2].getText());
  ASSERT_EQ(tok::newline,        Toks[3].getKind());

  ASSERT_EQ(tok::text,           Toks[4].getKind());
  ASSERT_EQ(StringRef(" Ccc"),   Toks[4].getText());
  ASSERT_EQ(tok::newline,        Toks[5].getKind());

  ASSERT_EQ(tok::text,           Toks[6].getKind());
  ASSERT_EQ(StringRef(" Ddd"),   Toks[6].getText());
  ASSERT_EQ(tok::newline,        Toks[7].getKind());
  ASSERT_EQ(tok::newline,        Toks[8].getKind());

  ASSERT_EQ(tok::text,           Toks[9].getKind());
  ASSERT_EQ(StringRef(" Eee"),   Toks[9].getText());

  ASSERT_EQ(tok::newline,        Toks[10].getKind());
  ASSERT_EQ(tok::newline,        Toks[11].getKind());
}

} // end namespace comments
} // end namespace clang

