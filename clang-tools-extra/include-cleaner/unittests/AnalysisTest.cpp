//===--- AnalysisTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
#include "AnalysisInternal.h"
#include "TypesInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace clang::include_cleaner {
namespace {
using testing::AllOf;
using testing::Contains;
using testing::ElementsAre;
using testing::Pair;
using testing::UnorderedElementsAre;

std::string guard(llvm::StringRef Code) {
  return "#pragma once\n" + Code.str();
}

class WalkUsedTest : public testing::Test {
protected:
  TestInputs Inputs;
  PragmaIncludes PI;
  WalkUsedTest() {
    Inputs.MakeAction = [this] {
      struct Hook : public SyntaxOnlyAction {
      public:
        Hook(PragmaIncludes *Out) : Out(Out) {}
        bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
          Out->record(CI);
          return true;
        }

        PragmaIncludes *Out;
      };
      return std::make_unique<Hook>(&PI);
    };
  }

  std::multimap<size_t, std::vector<Header>>
  offsetToProviders(TestAST &AST,
                    llvm::ArrayRef<SymbolReference> MacroRefs = {}) {
    const auto &SM = AST.sourceManager();
    llvm::SmallVector<Decl *> TopLevelDecls;
    for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
      if (!SM.isWrittenInMainFile(SM.getExpansionLoc(D->getLocation())))
        continue;
      TopLevelDecls.emplace_back(D);
    }
    std::multimap<size_t, std::vector<Header>> OffsetToProviders;
    walkUsed(TopLevelDecls, MacroRefs, &PI, AST.preprocessor(),
             [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
               auto [FID, Offset] = SM.getDecomposedLoc(Ref.RefLocation);
               if (FID != SM.getMainFileID())
                 ADD_FAILURE() << "Reference outside of the main file!";
               OffsetToProviders.emplace(Offset, Providers.vec());
             });
    return OffsetToProviders;
  }
};

TEST_F(WalkUsedTest, Basic) {
  llvm::Annotations Code(R"cpp(
  #include "header.h"
  #include "private.h"

  // No reference reported for the Parameter "p".
  void $bar^bar($private^Private p) {
    $foo^foo();
    std::$vector^vector $vconstructor^$v^v;
    $builtin^__builtin_popcount(1);
    std::$move^move(3);
  }
  )cpp");
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["header.h"] = guard(R"cpp(
  void foo();
  namespace std { class vector {}; int&& move(int&&); }
  )cpp");
  Inputs.ExtraFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "path/public.h"
    class Private {};
  )cpp");

  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  auto HeaderFile = Header(AST.fileManager().getFile("header.h").get());
  auto PrivateFile = Header(AST.fileManager().getFile("private.h").get());
  auto PublicFile = Header("\"path/public.h\"");
  auto MainFile = Header(SM.getFileEntryForID(SM.getMainFileID()));
  auto VectorSTL = Header(*tooling::stdlib::Header::named("<vector>"));
  auto UtilitySTL = Header(*tooling::stdlib::Header::named("<utility>"));
  EXPECT_THAT(
      offsetToProviders(AST),
      UnorderedElementsAre(
          Pair(Code.point("bar"), UnorderedElementsAre(MainFile)),
          Pair(Code.point("private"),
               UnorderedElementsAre(PublicFile, PrivateFile)),
          Pair(Code.point("foo"), UnorderedElementsAre(HeaderFile)),
          Pair(Code.point("vector"), UnorderedElementsAre(VectorSTL)),
          Pair(Code.point("vconstructor"), UnorderedElementsAre(VectorSTL)),
          Pair(Code.point("v"), UnorderedElementsAre(MainFile)),
          Pair(Code.point("builtin"), testing::IsEmpty()),
          Pair(Code.point("move"), UnorderedElementsAre(UtilitySTL))));
}

TEST_F(WalkUsedTest, MultipleProviders) {
  llvm::Annotations Code(R"cpp(
  #include "header1.h"
  #include "header2.h"
  void foo();

  void bar() {
    $foo^foo();
  }
  )cpp");
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["header1.h"] = guard(R"cpp(
  void foo();
  )cpp");
  Inputs.ExtraFiles["header2.h"] = guard(R"cpp(
  void foo();
  )cpp");

  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  auto HeaderFile1 = Header(AST.fileManager().getFile("header1.h").get());
  auto HeaderFile2 = Header(AST.fileManager().getFile("header2.h").get());
  auto MainFile = Header(SM.getFileEntryForID(SM.getMainFileID()));
  EXPECT_THAT(
      offsetToProviders(AST),
      Contains(Pair(Code.point("foo"),
                    UnorderedElementsAre(HeaderFile1, HeaderFile2, MainFile))));
}

TEST_F(WalkUsedTest, MacroRefs) {
  llvm::Annotations Code(R"cpp(
    #include "hdr.h"
    int $3^x = $1^ANSWER;
    int $4^y = $2^ANSWER;
  )cpp");
  llvm::Annotations Hdr(guard("#define ^ANSWER 42"));
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["hdr.h"] = Hdr.code();
  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  auto &PP = AST.preprocessor();
  const auto *HdrFile = SM.getFileManager().getFile("hdr.h").get();
  auto MainFile = Header(SM.getFileEntryForID(SM.getMainFileID()));

  auto HdrID = SM.translateFile(HdrFile);

  Symbol Answer1 = Macro{PP.getIdentifierInfo("ANSWER"),
                         SM.getComposedLoc(HdrID, Hdr.point())};
  Symbol Answer2 = Macro{PP.getIdentifierInfo("ANSWER"),
                         SM.getComposedLoc(HdrID, Hdr.point())};
  EXPECT_THAT(
      offsetToProviders(
          AST,
          {SymbolReference{
               Answer1, SM.getComposedLoc(SM.getMainFileID(), Code.point("1")),
               RefType::Explicit},
           SymbolReference{
               Answer2, SM.getComposedLoc(SM.getMainFileID(), Code.point("2")),
               RefType::Explicit}}),
      UnorderedElementsAre(
          Pair(Code.point("1"), UnorderedElementsAre(HdrFile)),
          Pair(Code.point("2"), UnorderedElementsAre(HdrFile)),
          Pair(Code.point("3"), UnorderedElementsAre(MainFile)),
          Pair(Code.point("4"), UnorderedElementsAre(MainFile))));
}

class AnalyzeTest : public testing::Test {
protected:
  TestInputs Inputs;
  PragmaIncludes PI;
  RecordedPP PP;
  AnalyzeTest() {
    Inputs.MakeAction = [this] {
      struct Hook : public SyntaxOnlyAction {
      public:
        Hook(RecordedPP &PP, PragmaIncludes &PI) : PP(PP), PI(PI) {}
        bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
          CI.getPreprocessor().addPPCallbacks(PP.record(CI.getPreprocessor()));
          PI.record(CI);
          return true;
        }

        RecordedPP &PP;
        PragmaIncludes &PI;
      };
      return std::make_unique<Hook>(PP, PI);
    };
  }
};

TEST_F(AnalyzeTest, Basic) {
  Inputs.Code = R"cpp(
#include "a.h"
#include "b.h"
#include "keep.h" // IWYU pragma: keep

int x = a + c;
)cpp";
  Inputs.ExtraFiles["a.h"] = guard("int a;");
  Inputs.ExtraFiles["b.h"] = guard(R"cpp(
    #include "c.h"
    int b;
  )cpp");
  Inputs.ExtraFiles["c.h"] = guard("int c;");
  Inputs.ExtraFiles["keep.h"] = guard("");
  TestAST AST(Inputs);
  auto Decls = AST.context().getTranslationUnitDecl()->decls();
  auto Results =
      analyze(std::vector<Decl *>{Decls.begin(), Decls.end()},
              PP.MacroReferences, PP.Includes, &PI, AST.preprocessor());

  const Include *B = PP.Includes.atLine(3);
  ASSERT_EQ(B->Spelled, "b.h");
  EXPECT_THAT(Results.Missing, ElementsAre("\"c.h\""));
  EXPECT_THAT(Results.Unused, ElementsAre(B));
}

TEST_F(AnalyzeTest, PrivateUsedInPublic) {
  // Check that umbrella header uses private include.
  Inputs.Code = R"cpp(#include "private.h")cpp";
  Inputs.ExtraFiles["private.h"] =
      guard("// IWYU pragma: private, include \"public.h\"");
  Inputs.FileName = "public.h";
  TestAST AST(Inputs);
  EXPECT_FALSE(PP.Includes.all().empty());
  auto Results = analyze({}, {}, PP.Includes, &PI, AST.preprocessor());
  EXPECT_THAT(Results.Unused, testing::IsEmpty());
}

TEST_F(AnalyzeTest, NoCrashWhenUnresolved) {
  // Check that umbrella header uses private include.
  Inputs.Code = R"cpp(#include "not_found.h")cpp";
  Inputs.ErrorOK = true;
  TestAST AST(Inputs);
  EXPECT_FALSE(PP.Includes.all().empty());
  auto Results = analyze({}, {}, PP.Includes, &PI, AST.preprocessor());
  EXPECT_THAT(Results.Unused, testing::IsEmpty());
}

TEST_F(AnalyzeTest, ResourceDirIsIgnored) {
  Inputs.ExtraArgs.push_back("-resource-dir");
  Inputs.ExtraArgs.push_back("resources");
  Inputs.ExtraArgs.push_back("-internal-isystem");
  Inputs.ExtraArgs.push_back("resources/include");
  Inputs.Code = R"cpp(
    #include <amintrin.h>
    #include <imintrin.h>
    void baz() {
      bar();
    }
  )cpp";
  Inputs.ExtraFiles["resources/include/amintrin.h"] = guard("");
  Inputs.ExtraFiles["resources/include/emintrin.h"] = guard(R"cpp(
    void bar();
  )cpp");
  Inputs.ExtraFiles["resources/include/imintrin.h"] = guard(R"cpp(
    #include <emintrin.h>
  )cpp");
  TestAST AST(Inputs);
  auto Results = analyze({}, {}, PP.Includes, &PI, AST.preprocessor());
  EXPECT_THAT(Results.Unused, testing::IsEmpty());
  EXPECT_THAT(Results.Missing, testing::IsEmpty());
}

TEST(FixIncludes, Basic) {
  llvm::StringRef Code = R"cpp(#include "d.h"
#include "a.h"
#include "b.h"
#include <c.h>
)cpp";

  Includes Inc;
  Include I;
  I.Spelled = "a.h";
  I.Line = 2;
  Inc.add(I);
  I.Spelled = "b.h";
  I.Line = 3;
  Inc.add(I);
  I.Spelled = "c.h";
  I.Line = 4;
  I.Angled = true;
  Inc.add(I);

  AnalysisResults Results;
  Results.Missing.push_back("\"aa.h\"");
  Results.Missing.push_back("\"ab.h\"");
  Results.Missing.push_back("<e.h>");
  Results.Unused.push_back(Inc.atLine(3));
  Results.Unused.push_back(Inc.atLine(4));

  EXPECT_EQ(fixIncludes(Results, "d.cc", Code, format::getLLVMStyle()),
R"cpp(#include "d.h"
#include "a.h"
#include "aa.h"
#include "ab.h"
#include <e.h>
)cpp");

  Results = {};
  Results.Missing.push_back("\"d.h\"");
  Code = R"cpp(#include "a.h")cpp";
  EXPECT_EQ(fixIncludes(Results, "d.cc", Code, format::getLLVMStyle()),
R"cpp(#include "d.h"
#include "a.h")cpp");
}

MATCHER_P3(expandedAt, FileID, Offset, SM, "") {
  auto [ExpanedFileID, ExpandedOffset] = SM->getDecomposedExpansionLoc(arg);
  return ExpanedFileID == FileID && ExpandedOffset == Offset;
}
MATCHER_P3(spelledAt, FileID, Offset, SM, "") {
  auto [SpelledFileID, SpelledOffset] = SM->getDecomposedSpellingLoc(arg);
  return SpelledFileID == FileID && SpelledOffset == Offset;
}
TEST(WalkUsed, FilterRefsNotSpelledInMainFile) {
  // Each test is expected to have a single expected ref of `target` symbol
  // (or have none).
  // The location in the reported ref is a macro location. $expand points to
  // the macro location, and $spell points to the spelled location.
  struct {
    llvm::StringRef Header;
    llvm::StringRef Main;
  } TestCases[] = {
      // Tests for decl references.
      {
          /*Header=*/"int target();",
          R"cpp(
            #define CALL_FUNC $spell^target()

            int b = $expand^CALL_FUNC;
          )cpp",
      },
      {/*Header=*/R"cpp(
           int target();
           #define CALL_FUNC target()
           )cpp",
       // No ref of `target` being reported, as it is not spelled in main file.
       "int a = CALL_FUNC;"},
      {
          /*Header=*/R"cpp(
            int target();
            #define PLUS_ONE(X) X() + 1
          )cpp",
          R"cpp(
            int a = $expand^PLUS_ONE($spell^target);
          )cpp",
      },
      {
          /*Header=*/R"cpp(
            int target();
            #define PLUS_ONE(X) X() + 1
          )cpp",
          R"cpp(
            int a = $expand^PLUS_ONE($spell^target);
          )cpp",
      },
      // Tests for macro references
      {/*Header=*/"#define target 1",
       R"cpp(
          #define USE_target $spell^target
          int b = $expand^USE_target;
        )cpp"},
      {/*Header=*/R"cpp(
          #define target 1
          #define USE_target target
        )cpp",
       // No ref of `target` being reported, it is not spelled in main file.
       R"cpp(
          int a = USE_target;
        )cpp"},
  };

  for (const auto &T : TestCases) {
    llvm::Annotations Main(T.Main);
    TestInputs Inputs(Main.code());
    Inputs.ExtraFiles["header.h"] = guard(T.Header);
    RecordedPP Recorded;
    Inputs.MakeAction = [&]() {
      struct RecordAction : public SyntaxOnlyAction {
        RecordedPP &Out;
        RecordAction(RecordedPP &Out) : Out(Out) {}
        bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
          auto &PP = CI.getPreprocessor();
          PP.addPPCallbacks(Out.record(PP));
          return true;
        }
      };
      return std::make_unique<RecordAction>(Recorded);
    };
    Inputs.ExtraArgs.push_back("-include");
    Inputs.ExtraArgs.push_back("header.h");
    TestAST AST(Inputs);
    llvm::SmallVector<Decl *> TopLevelDecls;
    for (Decl *D : AST.context().getTranslationUnitDecl()->decls())
      TopLevelDecls.emplace_back(D);
    auto &SM = AST.sourceManager();

    SourceLocation RefLoc;
    walkUsed(TopLevelDecls, Recorded.MacroReferences,
             /*PragmaIncludes=*/nullptr, AST.preprocessor(),
             [&](const SymbolReference &Ref, llvm::ArrayRef<Header>) {
               if (!Ref.RefLocation.isMacroID())
                 return;
               if (llvm::to_string(Ref.Target) == "target") {
                 ASSERT_TRUE(RefLoc.isInvalid())
                     << "Expected only one 'target' ref loc per testcase";
                 RefLoc = Ref.RefLocation;
               }
             });
    FileID MainFID = SM.getMainFileID();
    if (RefLoc.isValid()) {
      EXPECT_THAT(RefLoc, AllOf(expandedAt(MainFID, Main.point("expand"), &SM),
                                spelledAt(MainFID, Main.point("spell"), &SM)))
          << T.Main.str();
    } else {
      EXPECT_THAT(Main.points(), testing::IsEmpty());
    }
  }
}

struct Tag {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Tag &T) {
    return OS << "Anon Tag";
  }
};
TEST(Hints, Ordering) {
  auto Hinted = [](Hints Hints) {
    return clang::include_cleaner::Hinted<Tag>({}, Hints);
  };
  EXPECT_LT(Hinted(Hints::None), Hinted(Hints::CompleteSymbol));
  EXPECT_LT(Hinted(Hints::CompleteSymbol), Hinted(Hints::PublicHeader));
  EXPECT_LT(Hinted(Hints::PreferredHeader), Hinted(Hints::PublicHeader));
  EXPECT_LT(Hinted(Hints::CompleteSymbol | Hints::PreferredHeader),
            Hinted(Hints::PublicHeader));
}

// Test ast traversal & redecl selection end-to-end for templates, as explicit
// instantiations/specializations are not redecls of the primary template. We
// need to make sure we're selecting the right ones.
TEST_F(WalkUsedTest, TemplateDecls) {
  llvm::Annotations Code(R"cpp(
    #include "fwd.h"
    #include "def.h"
    #include "partial.h"
    template <> struct $exp_spec^Foo<char> {};
    template struct $exp^Foo<int>;
    $full^Foo<int> x;
    $implicit^Foo<bool> y;
    $partial^Foo<int*> z;
  )cpp");
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["fwd.h"] = guard("template<typename> struct Foo;");
  Inputs.ExtraFiles["def.h"] = guard("template<typename> struct Foo {};");
  Inputs.ExtraFiles["partial.h"] =
      guard("template<typename T> struct Foo<T*> {};");
  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  const auto *Fwd = SM.getFileManager().getFile("fwd.h").get();
  const auto *Def = SM.getFileManager().getFile("def.h").get();
  const auto *Partial = SM.getFileManager().getFile("partial.h").get();

  EXPECT_THAT(
      offsetToProviders(AST),
      AllOf(Contains(
                Pair(Code.point("exp_spec"), UnorderedElementsAre(Fwd, Def))),
            Contains(Pair(Code.point("exp"), UnorderedElementsAre(Fwd, Def))),
            Contains(Pair(Code.point("full"), UnorderedElementsAre(Fwd, Def))),
            Contains(
                Pair(Code.point("implicit"), UnorderedElementsAre(Fwd, Def))),
            Contains(
                Pair(Code.point("partial"), UnorderedElementsAre(Partial)))));
}

TEST_F(WalkUsedTest, IgnoresIdentityMacros) {
  llvm::Annotations Code(R"cpp(
  #include "header.h"
  void $bar^bar() {
    $stdin^stdin();
  }
  )cpp");
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["header.h"] = guard(R"cpp(
  #include "inner.h"
  void stdin();
  )cpp");
  Inputs.ExtraFiles["inner.h"] = guard(R"cpp(
  #define stdin stdin
  )cpp");

  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  auto MainFile = Header(SM.getFileEntryForID(SM.getMainFileID()));
  EXPECT_THAT(offsetToProviders(AST),
              UnorderedElementsAre(
                  // FIXME: we should have a reference from stdin to header.h
                  Pair(Code.point("bar"), UnorderedElementsAre(MainFile))));
}
} // namespace
} // namespace clang::include_cleaner
