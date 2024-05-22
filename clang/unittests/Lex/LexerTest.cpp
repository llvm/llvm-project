//===- unittests/Lex/LexerTest.cpp ------ Lexer tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Lexer.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace clang;
using testing::ElementsAre;

// The test fixture.
class LexerTest : public ::testing::Test {
protected:
  LexerTest()
    : FileMgr(FileMgrOpts),
      DiagID(new DiagnosticIDs()),
      Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
      SourceMgr(Diags, FileMgr),
      TargetOpts(new TargetOptions)
  {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  std::unique_ptr<Preprocessor> CreatePP(StringRef Source,
                                         TrivialModuleLoader &ModLoader) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(Source);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    std::unique_ptr<Preprocessor> PP = std::make_unique<Preprocessor>(
        std::make_shared<PreprocessorOptions>(), Diags, LangOpts, SourceMgr,
        HeaderInfo, ModLoader,
        /*IILookup =*/nullptr,
        /*OwnsHeaderSearch =*/false);
    PP->Initialize(*Target);
    PP->EnterMainSourceFile();
    return PP;
  }

  std::vector<Token> Lex(StringRef Source) {
    TrivialModuleLoader ModLoader;
    PP = CreatePP(Source, ModLoader);

    std::vector<Token> toks;
    PP->LexTokensUntilEOF(&toks);

    return toks;
  }

  std::vector<Token> CheckLex(StringRef Source,
                              ArrayRef<tok::TokenKind> ExpectedTokens) {
    auto toks = Lex(Source);
    EXPECT_EQ(ExpectedTokens.size(), toks.size());
    for (unsigned i = 0, e = ExpectedTokens.size(); i != e; ++i) {
      EXPECT_EQ(ExpectedTokens[i], toks[i].getKind());
    }

    return toks;
  }

  std::string getSourceText(Token Begin, Token End) {
    bool Invalid;
    StringRef Str =
        Lexer::getSourceText(CharSourceRange::getTokenRange(SourceRange(
                                    Begin.getLocation(), End.getLocation())),
                             SourceMgr, LangOpts, &Invalid);
    if (Invalid)
      return "<INVALID>";
    return std::string(Str);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
  std::unique_ptr<Preprocessor> PP;
};

TEST_F(LexerTest, GetSourceTextExpandsToMaximumInMacroArgument) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "M(f(M(i)))",
                                     ExpectedTokens);

  EXPECT_EQ("M(i)", getSourceText(toks[2], toks[2]));
}

TEST_F(LexerTest, GetSourceTextExpandsToMaximumInMacroArgumentForEndOfMacro) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "M(M(i) c)",
                                     ExpectedTokens);

  EXPECT_EQ("M(i)", getSourceText(toks[0], toks[0]));
}

TEST_F(LexerTest, GetSourceTextExpandsInMacroArgumentForBeginOfMacro) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "M(c c M(i))",
                                     ExpectedTokens);

  EXPECT_EQ("c M(i)", getSourceText(toks[1], toks[2]));
}

TEST_F(LexerTest, GetSourceTextExpandsInMacroArgumentForEndOfMacro) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "M(M(i) c c)",
                                     ExpectedTokens);

  EXPECT_EQ("M(i) c", getSourceText(toks[0], toks[1]));
}

TEST_F(LexerTest, GetSourceTextInSeparateFnMacros) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "M(c M(i)) M(M(i) c)",
                                     ExpectedTokens);

  EXPECT_EQ("<INVALID>", getSourceText(toks[1], toks[2]));
}

TEST_F(LexerTest, GetSourceTextWorksAcrossTokenPastes) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "#define C(x) M(x##c)\n"
                                     "M(f(C(i)))",
                                     ExpectedTokens);

  EXPECT_EQ("C(i)", getSourceText(toks[2], toks[2]));
}

TEST_F(LexerTest, GetSourceTextExpandsAcrossMultipleMacroCalls) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "f(M(M(i)))",
                                     ExpectedTokens);
  EXPECT_EQ("M(M(i))", getSourceText(toks[2], toks[2]));
}

TEST_F(LexerTest, GetSourceTextInMiddleOfMacroArgument) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "M(f(i))",
                                     ExpectedTokens);
  EXPECT_EQ("i", getSourceText(toks[2], toks[2]));
}

TEST_F(LexerTest, GetSourceTextExpandsAroundDifferentMacroCalls) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "#define C(x) x\n"
                                     "f(C(M(i)))",
                                     ExpectedTokens);
  EXPECT_EQ("C(M(i))", getSourceText(toks[2], toks[2]));
}

TEST_F(LexerTest, GetSourceTextOnlyExpandsIfFirstTokenInMacro) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "#define C(x) c x\n"
                                     "f(C(M(i)))",
                                     ExpectedTokens);
  EXPECT_EQ("M(i)", getSourceText(toks[3], toks[3]));
}

TEST_F(LexerTest, GetSourceTextExpandsRecursively) {
  std::vector<tok::TokenKind> ExpectedTokens;
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::l_paren);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_paren);

  std::vector<Token> toks = CheckLex("#define M(x) x\n"
                                     "#define C(x) c M(x)\n"
                                     "C(f(M(i)))",
                                     ExpectedTokens);
  EXPECT_EQ("M(i)", getSourceText(toks[3], toks[3]));
}

TEST_F(LexerTest, LexAPI) {
  std::vector<tok::TokenKind> ExpectedTokens;
  // Line 1 (after the #defines)
  ExpectedTokens.push_back(tok::l_square);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_square);
  ExpectedTokens.push_back(tok::l_square);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::r_square);
  // Line 2
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::identifier);

  std::vector<Token> toks = CheckLex("#define M(x) [x]\n"
                                     "#define N(x) x\n"
                                     "#define INN(x) x\n"
                                     "#define NOF1 INN(val)\n"
                                     "#define NOF2 val\n"
                                     "M(foo) N([bar])\n"
                                     "N(INN(val)) N(NOF1) N(NOF2) N(val)",
                                     ExpectedTokens);

  SourceLocation lsqrLoc = toks[0].getLocation();
  SourceLocation idLoc = toks[1].getLocation();
  SourceLocation rsqrLoc = toks[2].getLocation();
  CharSourceRange macroRange = SourceMgr.getExpansionRange(lsqrLoc);

  SourceLocation Loc;
  EXPECT_TRUE(Lexer::isAtStartOfMacroExpansion(lsqrLoc, SourceMgr, LangOpts, &Loc));
  EXPECT_EQ(Loc, macroRange.getBegin());
  EXPECT_FALSE(Lexer::isAtStartOfMacroExpansion(idLoc, SourceMgr, LangOpts));
  EXPECT_FALSE(Lexer::isAtEndOfMacroExpansion(idLoc, SourceMgr, LangOpts));
  EXPECT_TRUE(Lexer::isAtEndOfMacroExpansion(rsqrLoc, SourceMgr, LangOpts, &Loc));
  EXPECT_EQ(Loc, macroRange.getEnd());
  EXPECT_TRUE(macroRange.isTokenRange());

  CharSourceRange range = Lexer::makeFileCharRange(
           CharSourceRange::getTokenRange(lsqrLoc, idLoc), SourceMgr, LangOpts);
  EXPECT_TRUE(range.isInvalid());
  range = Lexer::makeFileCharRange(CharSourceRange::getTokenRange(idLoc, rsqrLoc),
                                   SourceMgr, LangOpts);
  EXPECT_TRUE(range.isInvalid());
  range = Lexer::makeFileCharRange(CharSourceRange::getTokenRange(lsqrLoc, rsqrLoc),
                                   SourceMgr, LangOpts);
  EXPECT_TRUE(!range.isTokenRange());
  EXPECT_EQ(range.getAsRange(),
            SourceRange(macroRange.getBegin(),
                        macroRange.getEnd().getLocWithOffset(1)));

  StringRef text = Lexer::getSourceText(
                               CharSourceRange::getTokenRange(lsqrLoc, rsqrLoc),
                               SourceMgr, LangOpts);
  EXPECT_EQ(text, "M(foo)");

  SourceLocation macroLsqrLoc = toks[3].getLocation();
  SourceLocation macroIdLoc = toks[4].getLocation();
  SourceLocation macroRsqrLoc = toks[5].getLocation();
  SourceLocation fileLsqrLoc = SourceMgr.getSpellingLoc(macroLsqrLoc);
  SourceLocation fileIdLoc = SourceMgr.getSpellingLoc(macroIdLoc);
  SourceLocation fileRsqrLoc = SourceMgr.getSpellingLoc(macroRsqrLoc);

  range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(macroLsqrLoc, macroIdLoc),
      SourceMgr, LangOpts);
  EXPECT_EQ(SourceRange(fileLsqrLoc, fileIdLoc.getLocWithOffset(3)),
            range.getAsRange());

  range = Lexer::makeFileCharRange(CharSourceRange::getTokenRange(macroIdLoc, macroRsqrLoc),
                                   SourceMgr, LangOpts);
  EXPECT_EQ(SourceRange(fileIdLoc, fileRsqrLoc.getLocWithOffset(1)),
            range.getAsRange());

  macroRange = SourceMgr.getExpansionRange(macroLsqrLoc);
  range = Lexer::makeFileCharRange(
                     CharSourceRange::getTokenRange(macroLsqrLoc, macroRsqrLoc),
                     SourceMgr, LangOpts);
  EXPECT_EQ(SourceRange(macroRange.getBegin(), macroRange.getEnd().getLocWithOffset(1)),
            range.getAsRange());

  text = Lexer::getSourceText(
          CharSourceRange::getTokenRange(SourceRange(macroLsqrLoc, macroIdLoc)),
          SourceMgr, LangOpts);
  EXPECT_EQ(text, "[bar");


  SourceLocation idLoc1 = toks[6].getLocation();
  SourceLocation idLoc2 = toks[7].getLocation();
  SourceLocation idLoc3 = toks[8].getLocation();
  SourceLocation idLoc4 = toks[9].getLocation();
  EXPECT_EQ("INN", Lexer::getImmediateMacroName(idLoc1, SourceMgr, LangOpts));
  EXPECT_EQ("INN", Lexer::getImmediateMacroName(idLoc2, SourceMgr, LangOpts));
  EXPECT_EQ("NOF2", Lexer::getImmediateMacroName(idLoc3, SourceMgr, LangOpts));
  EXPECT_EQ("N", Lexer::getImmediateMacroName(idLoc4, SourceMgr, LangOpts));
}

TEST_F(LexerTest, HandlesSplitTokens) {
  std::vector<tok::TokenKind> ExpectedTokens;
  // Line 1 (after the #defines)
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::less);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::less);
  ExpectedTokens.push_back(tok::greatergreater);
  // Line 2
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::less);
  ExpectedTokens.push_back(tok::identifier);
  ExpectedTokens.push_back(tok::less);
  ExpectedTokens.push_back(tok::greatergreater);

  std::vector<Token> toks = CheckLex("#define TY ty\n"
                                     "#define RANGLE ty<ty<>>\n"
                                     "TY<ty<>>\n"
                                     "RANGLE",
                                     ExpectedTokens);

  SourceLocation outerTyLoc = toks[0].getLocation();
  SourceLocation innerTyLoc = toks[2].getLocation();
  SourceLocation gtgtLoc = toks[4].getLocation();
  // Split the token to simulate the action of the parser and force creation of
  // an `ExpansionTokenRange`.
  SourceLocation rangleLoc = PP->SplitToken(gtgtLoc, 1);

  // Verify that it only captures the first greater-then and not the second one.
  CharSourceRange range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(innerTyLoc, rangleLoc), SourceMgr,
      LangOpts);
  EXPECT_TRUE(range.isCharRange());
  EXPECT_EQ(range.getAsRange(),
            SourceRange(innerTyLoc, gtgtLoc.getLocWithOffset(1)));

  // Verify case where range begins in a macro expansion.
  range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(outerTyLoc, rangleLoc), SourceMgr,
      LangOpts);
  EXPECT_TRUE(range.isCharRange());
  EXPECT_EQ(range.getAsRange(),
            SourceRange(SourceMgr.getExpansionLoc(outerTyLoc),
                        gtgtLoc.getLocWithOffset(1)));

  SourceLocation macroInnerTyLoc = toks[7].getLocation();
  SourceLocation macroGtgtLoc = toks[9].getLocation();
  // Split the token to simulate the action of the parser and force creation of
  // an `ExpansionTokenRange`.
  SourceLocation macroRAngleLoc = PP->SplitToken(macroGtgtLoc, 1);

  // Verify that it fails (because it only captures the first greater-then and
  // not the second one, so it doesn't span the entire macro expansion).
  range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(macroInnerTyLoc, macroRAngleLoc),
      SourceMgr, LangOpts);
  EXPECT_TRUE(range.isInvalid());
}

TEST_F(LexerTest, DontMergeMacroArgsFromDifferentMacroFiles) {
  std::vector<Token> toks =
      Lex("#define helper1 0\n"
          "void helper2(const char *, ...);\n"
          "#define M1(a, ...) helper2(a, ##__VA_ARGS__)\n"
          "#define M2(a, ...) M1(a, helper1, ##__VA_ARGS__)\n"
          "void f1() { M2(\"a\", \"b\"); }");

  // Check the file corresponding to the "helper1" macro arg in M2.
  //
  // The lexer used to report its size as 31, meaning that the end of the
  // expansion would be on the *next line* (just past `M2("a", "b")`). Make
  // sure that we get the correct end location (the comma after "helper1").
  SourceLocation helper1ArgLoc = toks[20].getLocation();
  EXPECT_EQ(SourceMgr.getFileIDSize(SourceMgr.getFileID(helper1ArgLoc)), 8U);
}

TEST_F(LexerTest, DontOverallocateStringifyArgs) {
  TrivialModuleLoader ModLoader;
  auto PP = CreatePP("\"StrArg\", 5, 'C'", ModLoader);

  llvm::BumpPtrAllocator Allocator;
  std::array<IdentifierInfo *, 3> ParamList;
  MacroInfo *MI = PP->AllocateMacroInfo({});
  MI->setIsFunctionLike();
  MI->setParameterList(ParamList, Allocator);
  EXPECT_EQ(3u, MI->getNumParams());
  EXPECT_TRUE(MI->isFunctionLike());

  Token Eof;
  Eof.setKind(tok::eof);
  std::vector<Token> ArgTokens;
  while (1) {
    Token tok;
    PP->Lex(tok);
    if (tok.is(tok::eof)) {
      ArgTokens.push_back(Eof);
      break;
    }
    if (tok.is(tok::comma))
      ArgTokens.push_back(Eof);
    else
      ArgTokens.push_back(tok);
  }

  auto MacroArgsDeleter = [&PP](MacroArgs *M) { M->destroy(*PP); };
  std::unique_ptr<MacroArgs, decltype(MacroArgsDeleter)> MA(
      MacroArgs::create(MI, ArgTokens, false, *PP), MacroArgsDeleter);
  auto StringifyArg = [&](int ArgNo) {
    return MA->StringifyArgument(MA->getUnexpArgument(ArgNo), *PP,
                                 /*Charify=*/false, {}, {});
  };
  Token Result = StringifyArg(0);
  EXPECT_EQ(tok::string_literal, Result.getKind());
  EXPECT_STREQ("\"\\\"StrArg\\\"\"", Result.getLiteralData());
  Result = StringifyArg(1);
  EXPECT_EQ(tok::string_literal, Result.getKind());
  EXPECT_STREQ("\"5\"", Result.getLiteralData());
  Result = StringifyArg(2);
  EXPECT_EQ(tok::string_literal, Result.getKind());
  EXPECT_STREQ("\"'C'\"", Result.getLiteralData());
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(StringifyArg(3), "Invalid arg #");
#endif
}

TEST_F(LexerTest, IsNewLineEscapedValid) {
  auto hasNewLineEscaped = [](const char *S) {
    return Lexer::isNewLineEscaped(S, S + strlen(S) - 1);
  };

  EXPECT_TRUE(hasNewLineEscaped("\\\r"));
  EXPECT_TRUE(hasNewLineEscaped("\\\n"));
  EXPECT_TRUE(hasNewLineEscaped("\\\r\n"));
  EXPECT_TRUE(hasNewLineEscaped("\\\n\r"));
  EXPECT_TRUE(hasNewLineEscaped("\\ \t\v\f\r"));
  EXPECT_TRUE(hasNewLineEscaped("\\ \t\v\f\r\n"));

  EXPECT_FALSE(hasNewLineEscaped("\\\r\r"));
  EXPECT_FALSE(hasNewLineEscaped("\\\r\r\n"));
  EXPECT_FALSE(hasNewLineEscaped("\\\n\n"));
  EXPECT_FALSE(hasNewLineEscaped("\r"));
  EXPECT_FALSE(hasNewLineEscaped("\n"));
  EXPECT_FALSE(hasNewLineEscaped("\r\n"));
  EXPECT_FALSE(hasNewLineEscaped("\n\r"));
  EXPECT_FALSE(hasNewLineEscaped("\r\r"));
  EXPECT_FALSE(hasNewLineEscaped("\n\n"));
}

TEST_F(LexerTest, GetBeginningOfTokenWithEscapedNewLine) {
  // Each line should have the same length for
  // further offset calculation to be more straightforward.
  const unsigned IdentifierLength = 8;
  std::string TextToLex = "rabarbar\n"
                          "foo\\\nbar\n"
                          "foo\\\rbar\n"
                          "fo\\\r\nbar\n"
                          "foo\\\n\rba\n";
  std::vector<tok::TokenKind> ExpectedTokens{5, tok::identifier};
  std::vector<Token> LexedTokens = CheckLex(TextToLex, ExpectedTokens);

  for (const Token &Tok : LexedTokens) {
    std::pair<FileID, unsigned> OriginalLocation =
        SourceMgr.getDecomposedLoc(Tok.getLocation());
    for (unsigned Offset = 0; Offset < IdentifierLength; ++Offset) {
      SourceLocation LookupLocation =
          Tok.getLocation().getLocWithOffset(Offset);

      std::pair<FileID, unsigned> FoundLocation =
          SourceMgr.getDecomposedExpansionLoc(
              Lexer::GetBeginningOfToken(LookupLocation, SourceMgr, LangOpts));

      // Check that location returned by the GetBeginningOfToken
      // is the same as original token location reported by Lexer.
      EXPECT_EQ(FoundLocation.second, OriginalLocation.second);
    }
  }
}

TEST_F(LexerTest, AvoidPastEndOfStringDereference) {
  EXPECT_TRUE(Lex("  //  \\\n").empty());
  EXPECT_TRUE(Lex("#include <\\\\").empty());
  EXPECT_TRUE(Lex("#include <\\\\\n").empty());
}

TEST_F(LexerTest, StringizingRasString) {
  // For "std::string Lexer::Stringify(StringRef Str, bool Charify)".
  std::string String1 = R"(foo
    {"bar":[]}
    baz)";
  // For "void Lexer::Stringify(SmallVectorImpl<char> &Str)".
  SmallString<128> String2;
  String2 += String1.c_str();

  // Corner cases.
  std::string String3 = R"(\
    \n
    \\n
    \\)";
  SmallString<128> String4;
  String4 += String3.c_str();
  std::string String5 = R"(a\


    \\b)";
  SmallString<128> String6;
  String6 += String5.c_str();

  String1 = Lexer::Stringify(StringRef(String1));
  Lexer::Stringify(String2);
  String3 = Lexer::Stringify(StringRef(String3));
  Lexer::Stringify(String4);
  String5 = Lexer::Stringify(StringRef(String5));
  Lexer::Stringify(String6);

  EXPECT_EQ(String1, R"(foo\n    {\"bar\":[]}\n    baz)");
  EXPECT_EQ(String2, R"(foo\n    {\"bar\":[]}\n    baz)");
  EXPECT_EQ(String3, R"(\\\n    \\n\n    \\\\n\n    \\\\)");
  EXPECT_EQ(String4, R"(\\\n    \\n\n    \\\\n\n    \\\\)");
  EXPECT_EQ(String5, R"(a\\\n\n\n    \\\\b)");
  EXPECT_EQ(String6, R"(a\\\n\n\n    \\\\b)");
}

TEST_F(LexerTest, CharRangeOffByOne) {
  std::vector<Token> toks = Lex(R"(#define MOO 1
    void foo() { MOO; })");
  const Token &moo = toks[5];

  EXPECT_EQ(getSourceText(moo, moo), "MOO");

  SourceRange R{moo.getLocation(), moo.getLocation()};

  EXPECT_TRUE(
      Lexer::isAtStartOfMacroExpansion(R.getBegin(), SourceMgr, LangOpts));
  EXPECT_TRUE(
      Lexer::isAtEndOfMacroExpansion(R.getEnd(), SourceMgr, LangOpts));

  CharSourceRange CR = Lexer::getAsCharRange(R, SourceMgr, LangOpts);

  EXPECT_EQ(Lexer::getSourceText(CR, SourceMgr, LangOpts), "MOO"); // Was "MO".
}

TEST_F(LexerTest, FindNextToken) {
  Lex("int abcd = 0;\n"
      "int xyz = abcd;\n");
  std::vector<std::string> GeneratedByNextToken;
  SourceLocation Loc =
      SourceMgr.getLocForStartOfFile(SourceMgr.getMainFileID());
  while (true) {
    auto T = Lexer::findNextToken(Loc, SourceMgr, LangOpts);
    ASSERT_TRUE(T);
    if (T->is(tok::eof))
      break;
    GeneratedByNextToken.push_back(getSourceText(*T, *T));
    Loc = T->getLocation();
  }
  EXPECT_THAT(GeneratedByNextToken, ElementsAre("abcd", "=", "0", ";", "int",
                                                "xyz", "=", "abcd", ";"));
}

TEST_F(LexerTest, CreatedFIDCountForPredefinedBuffer) {
  TrivialModuleLoader ModLoader;
  auto PP = CreatePP("", ModLoader);
  PP->LexTokensUntilEOF();
  EXPECT_EQ(SourceMgr.getNumCreatedFIDsForFileID(PP->getPredefinesFileID()),
            1U);
}

TEST_F(LexerTest, RawAndNormalLexSameForLineComments) {
  const llvm::StringLiteral Source = R"cpp(
  // First line comment.
  //* Second line comment which is ambigious.
  ; // Have a non-comment token to make sure something is lexed.
  )cpp";
  LangOpts.LineComment = false;
  auto Toks = Lex(Source);
  auto &SM = PP->getSourceManager();
  auto SrcBuffer = SM.getBufferData(SM.getMainFileID());
  Lexer L(SM.getLocForStartOfFile(SM.getMainFileID()), PP->getLangOpts(),
          SrcBuffer.data(), SrcBuffer.data(),
          SrcBuffer.data() + SrcBuffer.size());

  auto ToksView = llvm::ArrayRef(Toks);
  clang::Token T;
  EXPECT_FALSE(ToksView.empty());
  while (!L.LexFromRawLexer(T)) {
    ASSERT_TRUE(!ToksView.empty());
    EXPECT_EQ(T.getKind(), ToksView.front().getKind());
    ToksView = ToksView.drop_front();
  }
  EXPECT_TRUE(ToksView.empty());
}

TEST(LexerPreambleTest, PreambleBounds) {
  std::vector<std::string> Cases = {
      R"cc([[
        #include <foo>
        ]]int bar;
      )cc",
      R"cc([[
        #include <foo>
      ]])cc",
      R"cc([[
        // leading comment
        #include <foo>
        ]]// trailing comment
        int bar;
      )cc",
      R"cc([[
        module;
        #include <foo>
        ]]module bar;
        int x;
      )cc",
  };
  for (const auto& Case : Cases) {
    llvm::Annotations A(Case);
    clang::LangOptions LangOpts;
    LangOpts.CPlusPlusModules = true;
    auto Bounds = Lexer::ComputePreamble(A.code(), LangOpts);
    EXPECT_EQ(Bounds.Size, A.range().End) << Case;
  }
}

} // anonymous namespace
