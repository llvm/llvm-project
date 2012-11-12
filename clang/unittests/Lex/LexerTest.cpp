//===- unittests/Basic/LexerTest.cpp ------ Lexer tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Config/config.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

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
    Target = TargetInfo::CreateTargetInfo(Diags, *TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  IntrusiveRefCntPtr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

class VoidModuleLoader : public ModuleLoader {
  virtual Module *loadModule(SourceLocation ImportLoc, ModuleIdPath Path,
                             Module::NameVisibilityKind Visibility,
                             bool IsInclusionDirective) {
    return 0;
  }
};

TEST_F(LexerTest, LexAPI) {
  const char *source =
    "#define M(x) [x]\n"
    "#define N(x) x\n"
    "#define INN(x) x\n"
    "#define NOF1 INN(val)\n"
    "#define NOF2 val\n"
    "M(foo) N([bar])\n"
    "N(INN(val)) N(NOF1) N(NOF2) N(val)";

  MemoryBuffer *buf = MemoryBuffer::getMemBuffer(source);
  (void)SourceMgr.createMainFileIDForMemBuffer(buf);

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(new HeaderSearchOptions, FileMgr, Diags, LangOpts, 
                          Target.getPtr());
  Preprocessor PP(new PreprocessorOptions(), Diags, LangOpts, Target.getPtr(),
                  SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/ 0,
                  /*OwnsHeaderSearch =*/false,
                  /*DelayInitialization =*/ false);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(10U, toks.size());
  ASSERT_EQ(tok::l_square, toks[0].getKind());
  ASSERT_EQ(tok::identifier, toks[1].getKind());
  ASSERT_EQ(tok::r_square, toks[2].getKind());
  ASSERT_EQ(tok::l_square, toks[3].getKind());
  ASSERT_EQ(tok::identifier, toks[4].getKind());
  ASSERT_EQ(tok::r_square, toks[5].getKind());
  ASSERT_EQ(tok::identifier, toks[6].getKind());
  ASSERT_EQ(tok::identifier, toks[7].getKind());
  ASSERT_EQ(tok::identifier, toks[8].getKind());
  ASSERT_EQ(tok::identifier, toks[9].getKind());
  
  SourceLocation lsqrLoc = toks[0].getLocation();
  SourceLocation idLoc = toks[1].getLocation();
  SourceLocation rsqrLoc = toks[2].getLocation();
  std::pair<SourceLocation,SourceLocation>
    macroPair = SourceMgr.getExpansionRange(lsqrLoc);
  SourceRange macroRange = SourceRange(macroPair.first, macroPair.second);

  SourceLocation Loc;
  EXPECT_TRUE(Lexer::isAtStartOfMacroExpansion(lsqrLoc, SourceMgr, LangOpts, &Loc));
  EXPECT_EQ(Loc, macroRange.getBegin());
  EXPECT_FALSE(Lexer::isAtStartOfMacroExpansion(idLoc, SourceMgr, LangOpts));
  EXPECT_FALSE(Lexer::isAtEndOfMacroExpansion(idLoc, SourceMgr, LangOpts));
  EXPECT_TRUE(Lexer::isAtEndOfMacroExpansion(rsqrLoc, SourceMgr, LangOpts, &Loc));
  EXPECT_EQ(Loc, macroRange.getEnd());

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

  macroPair = SourceMgr.getExpansionRange(macroLsqrLoc);
  range = Lexer::makeFileCharRange(
                     CharSourceRange::getTokenRange(macroLsqrLoc, macroRsqrLoc),
                     SourceMgr, LangOpts);
  EXPECT_EQ(SourceRange(macroPair.first, macroPair.second.getLocWithOffset(1)),
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

} // anonymous namespace
