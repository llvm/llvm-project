//===--- PreambleTests.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "Compiler.h"
#include "Config.h"
#include "Diagnostics.h"
#include "Headers.h"
#include "Hover.h"
#include "ParsedAST.h"
#include "Preamble.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestFS.h"
#include "TestTU.h"
#include "XRefs.h"
#include "support/Context.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/PrecompiledPreamble.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest-matchers.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using testing::AllOf;
using testing::Contains;
using testing::ElementsAre;
using testing::Field;
using testing::IsEmpty;
using testing::Matcher;
using testing::MatchesRegex;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

namespace clang {
namespace clangd {
namespace {

MATCHER_P2(Distance, File, D, "") {
  return arg.first() == File && arg.second == D;
}

// Builds a preamble for BaselineContents, patches it for ModifiedContents and
// returns the includes in the patch.
IncludeStructure
collectPatchedIncludes(llvm::StringRef ModifiedContents,
                       llvm::StringRef BaselineContents,
                       llvm::StringRef MainFileName = "main.cpp") {
  MockFS FS;
  auto TU = TestTU::withCode(BaselineContents);
  TU.Filename = MainFileName.str();
  // ms-compatibility changes meaning of #import, make sure it is turned off.
  TU.ExtraArgs = {"-fno-ms-compatibility"};
  auto BaselinePreamble = TU.preamble();
  // Create the patch.
  TU.Code = ModifiedContents.str();
  auto PI = TU.inputs(FS);
  auto PP = PreamblePatch::createFullPatch(testPath(TU.Filename), PI,
                                           *BaselinePreamble);
  // Collect patch contents.
  IgnoreDiagnostics Diags;
  auto CI = buildCompilerInvocation(PI, Diags);
  PP.apply(*CI);
  // Run preprocessor over the modified contents with patched Invocation. We
  // provide a preamble and trim contents to ensure only the implicit header
  // introduced by the patch is parsed and nothing else.
  // We don't run PP directly over the patch cotents to test production
  // behaviour.
  auto Bounds = Lexer::ComputePreamble(ModifiedContents, CI->getLangOpts());
  auto Clang =
      prepareCompilerInstance(std::move(CI), &BaselinePreamble->Preamble,
                              llvm::MemoryBuffer::getMemBufferCopy(
                                  ModifiedContents.slice(0, Bounds.Size).str()),
                              PI.TFS->view(PI.CompileCommand.Directory), Diags);
  PreprocessOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0])) {
    ADD_FAILURE() << "failed begin source file";
    return {};
  }
  IncludeStructure Includes;
  Includes.collect(*Clang);
  if (llvm::Error Err = Action.Execute()) {
    ADD_FAILURE() << "failed to execute action: " << std::move(Err);
    return {};
  }
  Action.EndSourceFile();
  return Includes;
}

// Check preamble lexing logic by building an empty preamble and patching it
// with all the contents.
TEST(PreamblePatchTest, IncludeParsing) {
  // We expect any line with a point to show up in the patch.
  llvm::StringRef Cases[] = {
      // Only preamble
      R"cpp(^#include "a.h")cpp",
      // Both preamble and mainfile
      R"cpp(
        ^#include "a.h"
        garbage, finishes preamble
        #include "a.h")cpp",
      // Mixed directives
      R"cpp(
        ^#include "a.h"
        #pragma directive
        // some comments
        ^#include_next <a.h>
        #ifdef skipped
        ^#import "a.h"
        #endif)cpp",
      // Broken directives
      R"cpp(
        #include "a
        ^#include "a.h"
        #include <b
        ^#include <b.h>)cpp",
      // Directive is not part of preamble if it is not the token immediately
      // followed by the hash (#).
      R"cpp(
        ^#include "a.h"
        #/**/include <b.h>)cpp",
  };

  for (const auto &Case : Cases) {
    Annotations Test(Case);
    const auto Code = Test.code();
    SCOPED_TRACE(Code);

    auto Includes =
        collectPatchedIncludes(Code, /*BaselineContents=*/"").MainFileIncludes;
    auto Points = Test.points();
    ASSERT_EQ(Includes.size(), Points.size());
    for (size_t I = 0, E = Includes.size(); I != E; ++I)
      EXPECT_EQ(Includes[I].HashLine, Points[I].line);
  }
}

TEST(PreamblePatchTest, ContainsNewIncludes) {
  constexpr llvm::StringLiteral BaselineContents = R"cpp(
    #include <a.h>
    #include <b.h> // This will be removed
    #include <c.h>
  )cpp";
  constexpr llvm::StringLiteral ModifiedContents = R"cpp(
    #include <a.h>
    #include <c.h> // This has changed a line.
    #include <c.h> // This is a duplicate.
    #include <d.h> // This is newly introduced.
  )cpp";
  auto Includes = collectPatchedIncludes(ModifiedContents, BaselineContents)
                      .MainFileIncludes;
  EXPECT_THAT(Includes, ElementsAre(AllOf(Field(&Inclusion::Written, "<d.h>"),
                                          Field(&Inclusion::HashLine, 4))));
}

TEST(PreamblePatchTest, MainFileIsEscaped) {
  auto Includes = collectPatchedIncludes("#include <a.h>", "", "file\"name.cpp")
                      .MainFileIncludes;
  EXPECT_THAT(Includes, ElementsAre(AllOf(Field(&Inclusion::Written, "<a.h>"),
                                          Field(&Inclusion::HashLine, 0))));
}

TEST(PreamblePatchTest, PatchesPreambleIncludes) {
  MockFS FS;
  IgnoreDiagnostics Diags;
  auto TU = TestTU::withCode(R"cpp(
    #include "a.h" // IWYU pragma: keep
    #include "c.h"
    #ifdef FOO
    #include "d.h"
    #endif
  )cpp");
  TU.AdditionalFiles["a.h"] = "#include \"b.h\"";
  TU.AdditionalFiles["b.h"] = "";
  TU.AdditionalFiles["c.h"] = "";
  auto PI = TU.inputs(FS);
  auto BaselinePreamble = buildPreamble(
      TU.Filename, *buildCompilerInvocation(PI, Diags), PI, true, nullptr);
  // We drop c.h from modified and add a new header. Since the latter is patched
  // we should only get a.h in preamble includes. d.h shouldn't be part of the
  // preamble, as it's coming from a disabled region.
  TU.Code = R"cpp(
    #include "a.h"
    #include "b.h"
    #ifdef FOO
    #include "d.h"
    #endif
  )cpp";
  auto PP = PreamblePatch::createFullPatch(testPath(TU.Filename), TU.inputs(FS),
                                           *BaselinePreamble);
  // Only a.h should exists in the preamble, as c.h has been dropped and b.h was
  // newly introduced.
  EXPECT_THAT(
      PP.preambleIncludes(),
      ElementsAre(AllOf(
          Field(&Inclusion::Written, "\"a.h\""),
          Field(&Inclusion::Resolved, testPath("a.h")),
          Field(&Inclusion::HeaderID, testing::Not(testing::Eq(std::nullopt))),
          Field(&Inclusion::FileKind, SrcMgr::CharacteristicKind::C_User))));
}

std::optional<ParsedAST>
createPatchedAST(llvm::StringRef Baseline, llvm::StringRef Modified,
                 llvm::StringMap<std::string> AdditionalFiles = {}) {
  auto TU = TestTU::withCode(Baseline);
  TU.AdditionalFiles = std::move(AdditionalFiles);
  auto BaselinePreamble = TU.preamble();
  if (!BaselinePreamble) {
    ADD_FAILURE() << "Failed to build baseline preamble";
    return std::nullopt;
  }

  IgnoreDiagnostics Diags;
  MockFS FS;
  TU.Code = Modified.str();
  auto CI = buildCompilerInvocation(TU.inputs(FS), Diags);
  if (!CI) {
    ADD_FAILURE() << "Failed to build compiler invocation";
    return std::nullopt;
  }
  return ParsedAST::build(testPath(TU.Filename), TU.inputs(FS), std::move(CI),
                          {}, BaselinePreamble);
}

std::string getPreamblePatch(llvm::StringRef Baseline,
                             llvm::StringRef Modified) {
  auto BaselinePreamble = TestTU::withCode(Baseline).preamble();
  if (!BaselinePreamble) {
    ADD_FAILURE() << "Failed to build baseline preamble";
    return "";
  }
  MockFS FS;
  auto TU = TestTU::withCode(Modified);
  return PreamblePatch::createFullPatch(testPath("main.cpp"), TU.inputs(FS),
                                        *BaselinePreamble)
      .text()
      .str();
}

TEST(PreamblePatchTest, IncludesArePreserved) {
  llvm::StringLiteral Baseline = R"(//error-ok
#include <foo>
#include <bar>
)";
  llvm::StringLiteral Modified = R"(//error-ok
#include <foo>
#include <bar>
#define FOO)";

  auto Includes = createPatchedAST(Baseline, Modified.str())
                      ->getIncludeStructure()
                      .MainFileIncludes;
  EXPECT_TRUE(!Includes.empty());
  EXPECT_EQ(Includes, TestTU::withCode(Baseline)
                          .build()
                          .getIncludeStructure()
                          .MainFileIncludes);
}

TEST(PreamblePatchTest, Define) {
  // BAR should be defined while parsing the AST.
  struct {
    const char *const Contents;
    const char *const ExpectedPatch;
  } Cases[] = {
      {
          R"cpp(
        #define BAR
        [[BAR]])cpp",
          R"cpp(#line 0 ".*main.cpp"
#undef BAR
#line 2
#define         BAR
)cpp",
      },
      // multiline macro
      {
          R"cpp(
        #define BAR \

        [[BAR]])cpp",
          R"cpp(#line 0 ".*main.cpp"
#undef BAR
#line 2
#define         BAR
)cpp",
      },
      // multiline macro
      {
          R"cpp(
        #define \
                BAR
        [[BAR]])cpp",
          R"cpp(#line 0 ".*main.cpp"
#undef BAR
#line 3
#define         BAR
)cpp",
      },
  };

  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Contents);
    llvm::Annotations Modified(Case.Contents);
    EXPECT_THAT(getPreamblePatch("", Modified.code()),
                MatchesRegex(Case.ExpectedPatch));

    auto AST = createPatchedAST("", Modified.code());
    ASSERT_TRUE(AST);
    std::vector<llvm::Annotations::Range> MacroRefRanges;
    for (auto &M : AST->getMacros().MacroRefs) {
      for (auto &O : M.getSecond())
        MacroRefRanges.push_back({O.StartOffset, O.EndOffset});
    }
    EXPECT_THAT(MacroRefRanges, Contains(Modified.range()));
  }
}

TEST(PreamblePatchTest, OrderingPreserved) {
  llvm::StringLiteral Baseline = "#define BAR(X) X";
  Annotations Modified(R"cpp(
    #define BAR(X, Y) X Y
    #define BAR(X) X
    [[BAR]](int y);
  )cpp");

  llvm::StringLiteral ExpectedPatch(R"cpp(#line 0 ".*main.cpp"
#undef BAR
#line 2
#define     BAR\(X, Y\) X Y
#undef BAR
#line 3
#define     BAR\(X\) X
)cpp");
  EXPECT_THAT(getPreamblePatch(Baseline, Modified.code()),
              MatchesRegex(ExpectedPatch.str()));

  auto AST = createPatchedAST(Baseline, Modified.code());
  ASSERT_TRUE(AST);
}

TEST(PreamblePatchTest, LocateMacroAtWorks) {
  struct {
    const char *const Baseline;
    const char *const Modified;
  } Cases[] = {
      // Addition of new directive
      {
          "",
          R"cpp(
            #define $def^FOO
            $use^FOO)cpp",
      },
      // Available inside preamble section
      {
          "",
          R"cpp(
            #define $def^FOO
            #undef $use^FOO)cpp",
      },
      // Available after undef, as we don't patch those
      {
          "",
          R"cpp(
            #define $def^FOO
            #undef FOO
            $use^FOO)cpp",
      },
      // Identifier on a different line
      {
          "",
          R"cpp(
            #define \
              $def^FOO
            $use^FOO)cpp",
      },
      // In presence of comment tokens
      {
          "",
          R"cpp(
            #\
              define /* FOO */\
              /* FOO */ $def^FOO
            $use^FOO)cpp",
      },
      // Moved around
      {
          "#define FOO",
          R"cpp(
            #define BAR
            #define $def^FOO
            $use^FOO)cpp",
      },
  };
  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Modified);
    llvm::Annotations Modified(Case.Modified);
    auto AST = createPatchedAST(Case.Baseline, Modified.code());
    ASSERT_TRUE(AST);

    const auto &SM = AST->getSourceManager();
    auto *MacroTok = AST->getTokens().spelledTokenAt(
        SM.getComposedLoc(SM.getMainFileID(), Modified.point("use")));
    ASSERT_TRUE(MacroTok);

    auto FoundMacro = locateMacroAt(*MacroTok, AST->getPreprocessor());
    ASSERT_TRUE(FoundMacro);
    EXPECT_THAT(FoundMacro->Name, "FOO");

    auto MacroLoc = FoundMacro->NameLoc;
    EXPECT_EQ(SM.getFileID(MacroLoc), SM.getMainFileID());
    EXPECT_EQ(SM.getFileOffset(MacroLoc), Modified.point("def"));
  }
}

TEST(PreamblePatchTest, LocateMacroAtDeletion) {
  {
    // We don't patch deleted define directives, make sure we don't crash.
    llvm::StringLiteral Baseline = "#define FOO";
    llvm::Annotations Modified("^FOO");

    auto AST = createPatchedAST(Baseline, Modified.code());
    ASSERT_TRUE(AST);

    const auto &SM = AST->getSourceManager();
    auto *MacroTok = AST->getTokens().spelledTokenAt(
        SM.getComposedLoc(SM.getMainFileID(), Modified.point()));
    ASSERT_TRUE(MacroTok);

    auto FoundMacro = locateMacroAt(*MacroTok, AST->getPreprocessor());
    ASSERT_TRUE(FoundMacro);
    EXPECT_THAT(FoundMacro->Name, "FOO");
    auto HI =
        getHover(*AST, offsetToPosition(Modified.code(), Modified.point()),
                 format::getLLVMStyle(), nullptr);
    ASSERT_TRUE(HI);
    EXPECT_THAT(HI->Definition, testing::IsEmpty());
  }

  {
    // Offset is valid, but underlying text is different.
    llvm::StringLiteral Baseline = "#define FOO";
    Annotations Modified(R"cpp(#define BAR
    ^FOO")cpp");

    auto AST = createPatchedAST(Baseline, Modified.code());
    ASSERT_TRUE(AST);

    auto HI = getHover(*AST, Modified.point(), format::getLLVMStyle(), nullptr);
    ASSERT_TRUE(HI);
    EXPECT_THAT(HI->Definition, "#define BAR");
  }
}

MATCHER_P(referenceRangeIs, R, "") { return arg.Loc.range == R; }

TEST(PreamblePatchTest, RefsToMacros) {
  struct {
    const char *const Baseline;
    const char *const Modified;
  } Cases[] = {
      // Newly added
      {
          "",
          R"cpp(
            #define ^FOO
            ^[[FOO]])cpp",
      },
      // Moved around
      {
          "#define FOO",
          R"cpp(
            #define BAR
            #define ^FOO
            ^[[FOO]])cpp",
      },
      // Ref in preamble section
      {
          "",
          R"cpp(
            #define ^FOO
            #undef ^FOO)cpp",
      },
  };

  for (const auto &Case : Cases) {
    Annotations Modified(Case.Modified);
    auto AST = createPatchedAST("", Modified.code());
    ASSERT_TRUE(AST);

    const auto &SM = AST->getSourceManager();
    std::vector<Matcher<ReferencesResult::Reference>> ExpectedLocations;
    for (const auto &R : Modified.ranges())
      ExpectedLocations.push_back(referenceRangeIs(R));

    for (const auto &P : Modified.points()) {
      auto *MacroTok = AST->getTokens().spelledTokenAt(SM.getComposedLoc(
          SM.getMainFileID(),
          llvm::cantFail(positionToOffset(Modified.code(), P))));
      ASSERT_TRUE(MacroTok);
      EXPECT_THAT(findReferences(*AST, P, 0).References,
                  testing::ElementsAreArray(ExpectedLocations));
    }
  }
}

TEST(TranslatePreamblePatchLocation, Simple) {
  auto TU = TestTU::withHeaderCode(R"cpp(
    #line 3 "main.cpp"
    int foo();)cpp");
  // Presumed line/col needs to be valid in the main file.
  TU.Code = R"cpp(// line 1
    // line 2
    // line 3
    // line 4)cpp";
  TU.Filename = "main.cpp";
  TU.HeaderFilename = "__preamble_patch__.h";
  TU.ImplicitHeaderGuard = false;

  auto AST = TU.build();
  auto &SM = AST.getSourceManager();
  auto &ND = findDecl(AST, "foo");
  EXPECT_NE(SM.getFileID(ND.getLocation()), SM.getMainFileID());

  auto TranslatedLoc = translatePreamblePatchLocation(ND.getLocation(), SM);
  auto DecompLoc = SM.getDecomposedLoc(TranslatedLoc);
  EXPECT_EQ(DecompLoc.first, SM.getMainFileID());
  EXPECT_EQ(SM.getLineNumber(DecompLoc.first, DecompLoc.second), 3U);
}

TEST(PreamblePatch, ModifiedBounds) {
  struct {
    const char *const Baseline;
    const char *const Modified;
  } Cases[] = {
      // Size increased
      {
          "",
          R"cpp(
            #define FOO
            FOO)cpp",
      },
      // Stayed same
      {"#define FOO", "#define BAR"},
      // Got smaller
      {
          R"cpp(
            #define FOO
            #undef FOO)cpp",
          "#define FOO"},
  };

  for (const auto &Case : Cases) {
    auto TU = TestTU::withCode(Case.Baseline);
    auto BaselinePreamble = TU.preamble();
    ASSERT_TRUE(BaselinePreamble);

    Annotations Modified(Case.Modified);
    TU.Code = Modified.code().str();
    MockFS FS;
    auto PP = PreamblePatch::createFullPatch(testPath(TU.Filename),
                                             TU.inputs(FS), *BaselinePreamble);

    IgnoreDiagnostics Diags;
    auto CI = buildCompilerInvocation(TU.inputs(FS), Diags);
    ASSERT_TRUE(CI);

    const auto ExpectedBounds =
        Lexer::ComputePreamble(Case.Modified, CI->getLangOpts());
    EXPECT_EQ(PP.modifiedBounds().Size, ExpectedBounds.Size);
    EXPECT_EQ(PP.modifiedBounds().PreambleEndsAtStartOfLine,
              ExpectedBounds.PreambleEndsAtStartOfLine);
  }
}

TEST(PreamblePatch, MacroLoc) {
  llvm::StringLiteral Baseline = "\n#define MACRO 12\nint num = MACRO;";
  llvm::StringLiteral Modified = " \n#define MACRO 12\nint num = MACRO;";
  auto AST = createPatchedAST(Baseline, Modified);
  ASSERT_TRUE(AST);
}

TEST(PreamblePatch, NoopWhenNotRequested) {
  llvm::StringLiteral Baseline = "#define M\nint num = M;";
  llvm::StringLiteral Modified = "#define M\n#include <foo.h>\nint num = M;";
  auto TU = TestTU::withCode(Baseline);
  auto BaselinePreamble = TU.preamble();
  ASSERT_TRUE(BaselinePreamble);

  TU.Code = Modified.str();
  MockFS FS;
  auto PP = PreamblePatch::createMacroPatch(testPath(TU.Filename),
                                            TU.inputs(FS), *BaselinePreamble);
  EXPECT_TRUE(PP.text().empty());
}

::testing::Matcher<const Diag &>
withNote(::testing::Matcher<Note> NoteMatcher) {
  return Field(&Diag::Notes, ElementsAre(NoteMatcher));
}
MATCHER_P(Diag, Range, "Diag at " + llvm::to_string(Range)) {
  return arg.Range == Range;
}
MATCHER_P2(Diag, Range, Name,
           "Diag at " + llvm::to_string(Range) + " = [" + Name + "]") {
  return arg.Range == Range && arg.Name == Name;
}

TEST(PreamblePatch, DiagnosticsFromMainASTAreInRightPlace) {
  {
    Annotations Code("#define FOO");
    // Check with removals from preamble.
    Annotations NewCode("[[x]];/* error-ok */");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "missing_type_specifier")));
  }
  {
    // Check with additions to preamble.
    Annotations Code("#define FOO");
    Annotations NewCode(R"(
#define FOO
#define BAR
[[x]];/* error-ok */)");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "missing_type_specifier")));
  }
}

TEST(PreamblePatch, DiagnosticsToPreamble) {
  Config Cfg;
  Cfg.Diagnostics.UnusedIncludes = Config::IncludesPolicy::Strict;
  Cfg.Diagnostics.MissingIncludes = Config::IncludesPolicy::Strict;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));

  llvm::StringMap<std::string> AdditionalFiles;
  AdditionalFiles["foo.h"] = "#pragma once";
  AdditionalFiles["bar.h"] = "#pragma once";
  {
    Annotations Code(R"(
// Test comment
[[#include "foo.h"]])");
    // Check with removals from preamble.
    Annotations NewCode(R"([[#  include "foo.h"]])");
    auto AST = createPatchedAST(Code.code(), NewCode.code(), AdditionalFiles);
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "unused-includes")));
  }
  {
    // Check with additions to preamble.
    Annotations Code(R"(
// Test comment
[[#include "foo.h"]])");
    Annotations NewCode(R"(
$bar[[#include "bar.h"]]
// Test comment
$foo[[#include "foo.h"]])");
    auto AST = createPatchedAST(Code.code(), NewCode.code(), AdditionalFiles);
    EXPECT_THAT(
        AST->getDiagnostics(),
        UnorderedElementsAre(Diag(NewCode.range("bar"), "unused-includes"),
                             Diag(NewCode.range("foo"), "unused-includes")));
  }
  {
    Annotations Code("#define [[FOO]] 1\n");
    // Check ranges for notes.
    // This also makes sure we don't generate missing-include diagnostics
    // because macros are redefined in preamble-patch.
    Annotations NewCode(R"(#define BARXYZ 1
#define $foo1[[FOO]] 1
void foo();
#define $foo2[[FOO]] 2)");
    auto AST = createPatchedAST(Code.code(), NewCode.code(), AdditionalFiles);
    EXPECT_THAT(
        AST->getDiagnostics(),
        ElementsAre(AllOf(Diag(NewCode.range("foo2"), "-Wmacro-redefined"),
                          withNote(Diag(NewCode.range("foo1"))))));
  }
}

TEST(PreamblePatch, TranslatesDiagnosticsInPreamble) {
  {
    // Check with additions to preamble.
    Annotations Code("#include [[<foo>]]");
    Annotations NewCode(R"(
#define BAR
#include [[<foo>]])");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "pp_file_not_found")));
  }
  {
    // Check with removals from preamble.
    Annotations Code(R"(
#define BAR
#include [[<foo>]])");
    Annotations NewCode("#include [[<foo>]]");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "pp_file_not_found")));
  }
  {
    // Drop line with diags.
    Annotations Code("#include [[<foo>]]");
    Annotations NewCode("#define BAR\n#define BAZ\n");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(), IsEmpty());
  }
  {
    // Picks closest line in case of multiple alternatives.
    Annotations Code("#include [[<foo>]]");
    Annotations NewCode(R"(
#define BAR
#include [[<foo>]]
#define BAR
#include <foo>)");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "pp_file_not_found")));
  }
  {
    // Drop diag if line spelling has changed.
    Annotations Code("#include [[<foo>]]");
    Annotations NewCode(" # include <foo>");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(), IsEmpty());
  }
  {
    // Multiple lines.
    Annotations Code(R"(
#define BAR
#include [[<fo\
o>]])");
    Annotations NewCode(R"(#include [[<fo\
o>]])");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(),
                ElementsAre(Diag(NewCode.range(), "pp_file_not_found")));
  }
  {
    // Multiple lines with change.
    Annotations Code(R"(
#define BAR
#include <fox>
#include [[<fo\
o>]])");
    Annotations NewCode(R"(#include <fo\
x>)");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(), IsEmpty());
  }
  {
    // Preserves notes.
    Annotations Code(R"(
#define $note[[BAR]] 1
#define $main[[BAR]] 2)");
    Annotations NewCode(R"(
#define BAZ 0
#define $note[[BAR]] 1
#define BAZ 0
#define $main[[BAR]] 2)");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(
        AST->getDiagnostics(),
        ElementsAre(AllOf(Diag(NewCode.range("main"), "-Wmacro-redefined"),
                          withNote(Diag(NewCode.range("note"))))));
  }
  {
    // Preserves diag without note.
    Annotations Code(R"(
#define $note[[BAR]] 1
#define $main[[BAR]] 2)");
    Annotations NewCode(R"(
#define $main[[BAR]] 2)");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(
        AST->getDiagnostics(),
        ElementsAre(AllOf(Diag(NewCode.range("main"), "-Wmacro-redefined"),
                          Field(&Diag::Notes, IsEmpty()))));
  }
  {
    // Make sure orphaned notes are not promoted to diags.
    Annotations Code(R"(
#define $note[[BAR]] 1
#define $main[[BAR]] 2)");
    Annotations NewCode(R"(
#define BAZ 0
#define BAR 1)");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(), IsEmpty());
  }
  {
    Annotations Code(R"(
#ifndef FOO
#define FOO
void foo();
#endif)");
    // This code will emit a diagnostic for unterminated #ifndef (as stale
    // preamble has the conditional but main file doesn't terminate it).
    // We shouldn't emit any diagnotiscs (and shouldn't crash).
    Annotations NewCode("");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getDiagnostics(), IsEmpty());
  }
  {
    Annotations Code(R"(
#ifndef FOO
#define FOO
void foo();
#endif)");
    // This code will emit a diagnostic for unterminated #ifndef (as stale
    // preamble has the conditional but main file doesn't terminate it).
    // We shouldn't emit any diagnotiscs (and shouldn't crash).
    // FIXME: Patch/ignore diagnostics in such cases.
    Annotations NewCode(R"(
i[[nt]] xyz;
    )");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(
        AST->getDiagnostics(),
        ElementsAre(Diag(NewCode.range(), "pp_unterminated_conditional")));
  }
}

MATCHER_P2(Mark, Range, Text, "") {
  return std::tie(arg.Rng, arg.Trivia) == std::tie(Range, Text);
}

TEST(PreamblePatch, MacroAndMarkHandling) {
  {
    Annotations Code(R"cpp(
#ifndef FOO
#define FOO
// Some comments
#pragma mark XX
#define BAR

#endif)cpp");
    Annotations NewCode(R"cpp(
#ifndef FOO
#define FOO
#define BAR
#pragma $x[[mark XX
]]
#pragma $y[[mark YY
]]
#define BAZ

#endif)cpp");
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    EXPECT_THAT(AST->getMacros().Names.keys(),
                UnorderedElementsAreArray({"FOO", "BAR", "BAZ"}));
    EXPECT_THAT(AST->getMarks(),
                UnorderedElementsAre(Mark(NewCode.range("x"), " XX"),
                                     Mark(NewCode.range("y"), " YY")));
  }
}

TEST(PreamblePatch, PatchFileEntry) {
  Annotations Code(R"cpp(#define FOO)cpp");
  Annotations NewCode(R"cpp(
#define BAR
#define FOO)cpp");
  {
    auto AST = createPatchedAST(Code.code(), Code.code());
    EXPECT_EQ(
        PreamblePatch::getPatchEntry(AST->tuPath(), AST->getSourceManager()),
        nullptr);
  }
  {
    auto AST = createPatchedAST(Code.code(), NewCode.code());
    auto FE =
        PreamblePatch::getPatchEntry(AST->tuPath(), AST->getSourceManager());
    ASSERT_NE(FE, std::nullopt);
    EXPECT_THAT(FE->getName().str(),
                testing::EndsWith(PreamblePatch::HeaderName.str()));
  }
}

} // namespace
} // namespace clangd
} // namespace clang
