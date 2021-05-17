//===- unittests/Analysis/MacroExpansionContextTest.cpp - -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/MacroExpansionContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

// static bool HACK_EnableDebugInUnitTest = (::llvm::DebugFlag = true);

namespace clang {
namespace analysis {
namespace {

class MacroExpansionContextTest : public ::testing::Test {
protected:
  MacroExpansionContextTest()
      : InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem),
        FileMgr(FileSystemOptions(), InMemoryFileSystem),
        DiagID(new DiagnosticIDs()), DiagOpts(new DiagnosticOptions()),
        Diags(DiagID, DiagOpts.get(), new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions()) {
    TargetOpts->Triple = "x86_64-pc-linux-unknown";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
    LangOpts.CPlusPlus20 = 1; // For __VA_OPT__
  }

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;

  std::unique_ptr<MacroExpansionContext>
  getMacroExpansionContextFor(StringRef SourceText) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(SourceText);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));
    TrivialModuleLoader ModLoader;
    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags, LangOpts,
                    SourceMgr, HeaderInfo, ModLoader,
                    /*IILookup =*/nullptr,
                    /*OwnsHeaderSearch =*/false);

    PP.Initialize(*Target);
    auto Ctx = std::make_unique<MacroExpansionContext>(LangOpts);
    Ctx->registerForPreprocessor(PP);

    // Lex source text.
    PP.EnterMainSourceFile();

    while (true) {
      Token Tok;
      PP.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
    }

    // Callbacks have been executed at this point.
    return Ctx;
  }

  /// Returns the expansion location to main file at the given row and column.
  SourceLocation at(unsigned row, unsigned col) const {
    SourceLocation Loc =
        SourceMgr.translateLineCol(SourceMgr.getMainFileID(), row, col);
    return SourceMgr.getExpansionLoc(Loc);
  }

  static std::string dumpExpandedTexts(const MacroExpansionContext &Ctx) {
    std::string Buf;
    llvm::raw_string_ostream OS{Buf};
    Ctx.dumpExpandedTextsToStream(OS);
    return OS.str();
  }

  static std::string dumpExpansionRanges(const MacroExpansionContext &Ctx) {
    std::string Buf;
    llvm::raw_string_ostream OS{Buf};
    Ctx.dumpExpansionRangesToStream(OS);
    return OS.str();
  }
};

TEST_F(MacroExpansionContextTest, IgnoresPragmas) {
  // No-crash during lexing.
  const auto Ctx = getMacroExpansionContextFor(R"code(
  _Pragma("pack(push, 1)")
  _Pragma("pack(pop, 1)")
      )code");
  // After preprocessing:
  // #pragma pack(push, 1)
  // #pragma pack(pop, 1)

  EXPECT_EQ("\n=============== ExpandedTokens ===============\n",
            dumpExpandedTexts(*Ctx));
  EXPECT_EQ("\n=============== ExpansionRanges ===============\n",
            dumpExpansionRanges(*Ctx));

  EXPECT_FALSE(Ctx->getExpandedText(at(2, 1)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(2, 1)).hasValue());

  EXPECT_FALSE(Ctx->getExpandedText(at(2, 3)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(2, 3)).hasValue());

  EXPECT_FALSE(Ctx->getExpandedText(at(3, 3)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(3, 3)).hasValue());
}

TEST_F(MacroExpansionContextTest, NoneForNonExpansionLocations) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define EMPTY
  A b cd EMPTY ef EMPTY gh
EMPTY zz
      )code");
  // After preprocessing:
  //  A b cd ef gh
  //      zz

  // That's the beginning of the definition of EMPTY.
  EXPECT_FALSE(Ctx->getExpandedText(at(2, 11)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(2, 11)).hasValue());

  // The space before the first expansion of EMPTY.
  EXPECT_FALSE(Ctx->getExpandedText(at(3, 9)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(3, 9)).hasValue());

  // The beginning of the first expansion of EMPTY.
  EXPECT_TRUE(Ctx->getExpandedText(at(3, 10)).hasValue());
  EXPECT_TRUE(Ctx->getOriginalText(at(3, 10)).hasValue());

  // Pointing inside of the token EMPTY, but not at the beginning.
  // FIXME: We only deal with begin locations.
  EXPECT_FALSE(Ctx->getExpandedText(at(3, 11)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(3, 11)).hasValue());

  // Same here.
  EXPECT_FALSE(Ctx->getExpandedText(at(3, 12)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(3, 12)).hasValue());

  // The beginning of the last expansion of EMPTY.
  EXPECT_TRUE(Ctx->getExpandedText(at(4, 1)).hasValue());
  EXPECT_TRUE(Ctx->getOriginalText(at(4, 1)).hasValue());

  // Same as for the 3:11 case.
  EXPECT_FALSE(Ctx->getExpandedText(at(4, 2)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(4, 2)).hasValue());
}

TEST_F(MacroExpansionContextTest, EmptyExpansions) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define EMPTY
  A b cd EMPTY ef EMPTY gh
EMPTY zz
      )code");
  // After preprocessing:
  //  A b cd ef gh
  //      zz

  EXPECT_EQ("", Ctx->getExpandedText(at(3, 10)).getValue());
  EXPECT_EQ("EMPTY", Ctx->getOriginalText(at(3, 10)).getValue());

  EXPECT_EQ("", Ctx->getExpandedText(at(3, 19)).getValue());
  EXPECT_EQ("EMPTY", Ctx->getOriginalText(at(3, 19)).getValue());

  EXPECT_EQ("", Ctx->getExpandedText(at(4, 1)).getValue());
  EXPECT_EQ("EMPTY", Ctx->getOriginalText(at(4, 1)).getValue());
}

TEST_F(MacroExpansionContextTest, TransitiveExpansions) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define EMPTY
  #define WOOF EMPTY ) EMPTY   1
  A b cd WOOF ef EMPTY gh
      )code");
  // After preprocessing:
  //  A b cd ) 1 ef gh

  EXPECT_EQ("WOOF", Ctx->getOriginalText(at(4, 10)).getValue());

  EXPECT_EQ("", Ctx->getExpandedText(at(4, 18)).getValue());
  EXPECT_EQ("EMPTY", Ctx->getOriginalText(at(4, 18)).getValue());
}

TEST_F(MacroExpansionContextTest, MacroFunctions) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define EMPTY
  #define WOOF(x) x(EMPTY ) )  ) EMPTY   1
  A b cd WOOF($$ ef) EMPTY gh
  WOOF(WOOF)
  WOOF(WOOF(bar barr))),,),')
      )code");
  // After preprocessing:
  //  A b cd $$ ef( ) ) ) 1 gh
  //  WOOF( ) ) ) 1
  //  bar barr( ) ) ) 1( ) ) ) 1),,),')

  EXPECT_EQ("$$ ef ()))1", Ctx->getExpandedText(at(4, 10)).getValue());
  EXPECT_EQ("WOOF($$ ef)", Ctx->getOriginalText(at(4, 10)).getValue());

  EXPECT_EQ("", Ctx->getExpandedText(at(4, 22)).getValue());
  EXPECT_EQ("EMPTY", Ctx->getOriginalText(at(4, 22)).getValue());

  EXPECT_EQ("WOOF ()))1", Ctx->getExpandedText(at(5, 3)).getValue());
  EXPECT_EQ("WOOF(WOOF)", Ctx->getOriginalText(at(5, 3)).getValue());

  EXPECT_EQ("bar barr ()))1()))1", Ctx->getExpandedText(at(6, 3)).getValue());
  EXPECT_EQ("WOOF(WOOF(bar barr))", Ctx->getOriginalText(at(6, 3)).getValue());
}

TEST_F(MacroExpansionContextTest, VariadicMacros) {
  // From the GCC website.
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define eprintf(format, ...) fprintf (stderr, format, __VA_ARGS__)
  eprintf("success!\n", );
  eprintf("success!\n");

  #define eprintf2(format, ...) \
    fprintf (stderr, format __VA_OPT__(,) __VA_ARGS__)
  eprintf2("success!\n", );
  eprintf2("success!\n");
      )code");
  // After preprocessing:
  //  fprintf (stderr, "success!\n", );
  //  fprintf (stderr, "success!\n", );
  //  fprintf (stderr, "success!\n" );
  //  fprintf (stderr, "success!\n" );

  EXPECT_EQ(R"(fprintf (stderr ,"success!\n",))",
            Ctx->getExpandedText(at(3, 3)).getValue());
  EXPECT_EQ(R"(eprintf("success!\n", ))",
            Ctx->getOriginalText(at(3, 3)).getValue());

  EXPECT_EQ(R"(fprintf (stderr ,"success!\n",))",
            Ctx->getExpandedText(at(4, 3)).getValue());
  EXPECT_EQ(R"(eprintf("success!\n"))",
            Ctx->getOriginalText(at(4, 3)).getValue());

  EXPECT_EQ(R"(fprintf (stderr ,"success!\n"))",
            Ctx->getExpandedText(at(8, 3)).getValue());
  EXPECT_EQ(R"(eprintf2("success!\n", ))",
            Ctx->getOriginalText(at(8, 3)).getValue());

  EXPECT_EQ(R"(fprintf (stderr ,"success!\n"))",
            Ctx->getExpandedText(at(9, 3)).getValue());
  EXPECT_EQ(R"(eprintf2("success!\n"))",
            Ctx->getOriginalText(at(9, 3)).getValue());
}

TEST_F(MacroExpansionContextTest, ConcatenationMacros) {
  // From the GCC website.
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define COMMAND(NAME)  { #NAME, NAME ## _command }
  struct command commands[] = {
    COMMAND(quit),
    COMMAND(help),
  };)code");
  // After preprocessing:
  //  struct command commands[] = {
  //    { "quit", quit_command },
  //    { "help", help_command },
  //  };

  EXPECT_EQ(R"({"quit",quit_command })",
            Ctx->getExpandedText(at(4, 5)).getValue());
  EXPECT_EQ("COMMAND(quit)", Ctx->getOriginalText(at(4, 5)).getValue());

  EXPECT_EQ(R"({"help",help_command })",
            Ctx->getExpandedText(at(5, 5)).getValue());
  EXPECT_EQ("COMMAND(help)", Ctx->getOriginalText(at(5, 5)).getValue());
}

TEST_F(MacroExpansionContextTest, StringizingMacros) {
  // From the GCC website.
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define WARN_IF(EXP) \
  do { if (EXP) \
          fprintf (stderr, "Warning: " #EXP "\n"); } \
  while (0)
  WARN_IF (x == 0);

  #define xstr(s) str(s)
  #define str(s) #s
  #define foo 4
  str (foo)
  xstr (foo)
      )code");
  // After preprocessing:
  //  do { if (x == 0) fprintf (stderr, "Warning: " "x == 0" "\n"); } while (0);
  //  "foo"
  //  "4"

  EXPECT_EQ(
      R"(do {if (x ==0)fprintf (stderr ,"Warning: ""x == 0""\n");}while (0))",
      Ctx->getExpandedText(at(6, 3)).getValue());
  EXPECT_EQ("WARN_IF (x == 0)", Ctx->getOriginalText(at(6, 3)).getValue());

  EXPECT_EQ(R"("foo")", Ctx->getExpandedText(at(11, 3)).getValue());
  EXPECT_EQ("str (foo)", Ctx->getOriginalText(at(11, 3)).getValue());

  EXPECT_EQ(R"("4")", Ctx->getExpandedText(at(12, 3)).getValue());
  EXPECT_EQ("xstr (foo)", Ctx->getOriginalText(at(12, 3)).getValue());
}

TEST_F(MacroExpansionContextTest, StringizingVariadicMacros) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define xstr(...) str(__VA_ARGS__)
  #define str(...) #__VA_ARGS__
  #define RParen2x ) )
  #define EMPTY
  #define f(x, ...) __VA_ARGS__ ! x * x
  #define g(...) zz EMPTY f(__VA_ARGS__ ! x) f() * y
  #define h(x, G) G(x) G(x ## x RParen2x
  #define q(G) h(apple, G(apple)) RParen2x

  q(g)
  q(xstr)
  g(RParen2x)
  f( RParen2x )s
      )code");
  // clang-format off
  // After preprocessing:
  //  zz ! apple ! x * apple ! x ! * * y(apple) zz ! apple ! x * apple ! x ! * * y(appleapple ) ) ) )
  //  "apple"(apple) "apple"(appleapple ) ) ) )
  //  zz ! * ) ! x) ! * * y
  //  ! ) ) * ) )
  // clang-format on

  EXPECT_EQ("zz !apple !x *apple !x !**y (apple )zz !apple !x *apple !x !**y "
            "(appleapple ))))",
            Ctx->getExpandedText(at(11, 3)).getValue());
  EXPECT_EQ("q(g)", Ctx->getOriginalText(at(11, 3)).getValue());

  EXPECT_EQ(R"res("apple"(apple )"apple"(appleapple )))))res",
            Ctx->getExpandedText(at(12, 3)).getValue());
  EXPECT_EQ("q(xstr)", Ctx->getOriginalText(at(12, 3)).getValue());

  EXPECT_EQ("zz !*)!x )!**y ", Ctx->getExpandedText(at(13, 3)).getValue());
  EXPECT_EQ("g(RParen2x)", Ctx->getOriginalText(at(13, 3)).getValue());

  EXPECT_EQ("!))*))", Ctx->getExpandedText(at(14, 3)).getValue());
  EXPECT_EQ("f( RParen2x )", Ctx->getOriginalText(at(14, 3)).getValue());
}

TEST_F(MacroExpansionContextTest, RedefUndef) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define Hi(x) Welcome x
  Hi(Adam)
  #define Hi Willkommen
  Hi Hans
  #undef Hi
  Hi(Hi)
      )code");
  // After preprocessing:
  //  Welcome Adam
  //  Willkommen Hans
  //  Hi(Hi)

  // FIXME: Extra space follows every identifier.
  EXPECT_EQ("Welcome Adam ", Ctx->getExpandedText(at(3, 3)).getValue());
  EXPECT_EQ("Hi(Adam)", Ctx->getOriginalText(at(3, 3)).getValue());

  EXPECT_EQ("Willkommen ", Ctx->getExpandedText(at(5, 3)).getValue());
  EXPECT_EQ("Hi", Ctx->getOriginalText(at(5, 3)).getValue());

  // There was no macro expansion at 7:3, we should expect None.
  EXPECT_FALSE(Ctx->getExpandedText(at(7, 3)).hasValue());
  EXPECT_FALSE(Ctx->getOriginalText(at(7, 3)).hasValue());
}

TEST_F(MacroExpansionContextTest, UnbalacedParenthesis) {
  const auto Ctx = getMacroExpansionContextFor(R"code(
  #define retArg(x) x
  #define retArgUnclosed retArg(fun()
  #define BB CC
  #define applyInt BB(int)
  #define CC(x) retArgUnclosed

  applyInt );

  #define expandArgUnclosedCommaExpr(x) (x, fun(), 1
  #define f expandArgUnclosedCommaExpr

  int x =  f(f(1))  ));
      )code");
  // After preprocessing:
  //  fun();
  //  int x = ((1, fun(), 1, fun(), 1 ));

  EXPECT_EQ("fun ()", Ctx->getExpandedText(at(8, 3)).getValue());
  EXPECT_EQ("applyInt )", Ctx->getOriginalText(at(8, 3)).getValue());

  EXPECT_EQ("((1,fun (),1,fun (),1",
            Ctx->getExpandedText(at(13, 12)).getValue());
  EXPECT_EQ("f(f(1))", Ctx->getOriginalText(at(13, 12)).getValue());
}

} // namespace
} // namespace analysis
} // namespace clang
