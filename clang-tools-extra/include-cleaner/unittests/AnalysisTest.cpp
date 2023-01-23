//===--- AnalysisTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
#include "AnalysisInternal.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>

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

  llvm::DenseMap<size_t, std::vector<Header>>
  offsetToProviders(TestAST &AST, SourceManager &SM,
                    llvm::ArrayRef<SymbolReference> MacroRefs = {}) {
    llvm::SmallVector<Decl *> TopLevelDecls;
    for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
      if (!SM.isWrittenInMainFile(SM.getExpansionLoc(D->getLocation())))
        continue;
      TopLevelDecls.emplace_back(D);
    }
    llvm::DenseMap<size_t, std::vector<Header>> OffsetToProviders;
    walkUsed(TopLevelDecls, MacroRefs, &PI, SM,
             [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
               auto [FID, Offset] = SM.getDecomposedLoc(Ref.RefLocation);
               if (FID != SM.getMainFileID())
                 ADD_FAILURE() << "Reference outside of the main file!";
               OffsetToProviders.try_emplace(Offset, Providers.vec());
             });
    return OffsetToProviders;
  }
};

TEST_F(WalkUsedTest, Basic) {
  llvm::Annotations Code(R"cpp(
  #include "header.h"
  #include "private.h"

  void $bar^bar($private^Private) {
    $foo^foo();
    std::$vector^vector $vconstructor^v;
  }
  )cpp");
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["header.h"] = guard(R"cpp(
  void foo();
  namespace std { class vector {}; }
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
  EXPECT_THAT(
      offsetToProviders(AST, SM),
      UnorderedElementsAre(
          Pair(Code.point("bar"), UnorderedElementsAre(MainFile)),
          Pair(Code.point("private"),
               UnorderedElementsAre(PublicFile, PrivateFile)),
          Pair(Code.point("foo"), UnorderedElementsAre(HeaderFile)),
          Pair(Code.point("vector"), UnorderedElementsAre(VectorSTL)),
          Pair(Code.point("vconstructor"), UnorderedElementsAre(VectorSTL))));
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
      offsetToProviders(AST, SM),
      Contains(Pair(Code.point("foo"),
                    UnorderedElementsAre(HeaderFile1, HeaderFile2, MainFile))));
}

TEST_F(WalkUsedTest, MacroRefs) {
  llvm::Annotations Code(R"cpp(
    #include "hdr.h"
    int x = $1^ANSWER;
    int y = $2^ANSWER;
  )cpp");
  llvm::Annotations Hdr(guard("#define ^ANSWER 42"));
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["hdr.h"] = Hdr.code();
  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  const auto *HdrFile = SM.getFileManager().getFile("hdr.h").get();
  auto HdrID = SM.translateFile(HdrFile);

  IdentifierTable Idents;
  Symbol Answer1 =
      Macro{&Idents.get("ANSWER"), SM.getComposedLoc(HdrID, Hdr.point())};
  Symbol Answer2 =
      Macro{&Idents.get("ANSWER"), SM.getComposedLoc(HdrID, Hdr.point())};
  EXPECT_THAT(
      offsetToProviders(AST, SM,
                        {SymbolReference{SM.getComposedLoc(SM.getMainFileID(),
                                                           Code.point("1")),
                                         Answer1, RefType::Explicit},
                         SymbolReference{SM.getComposedLoc(SM.getMainFileID(),
                                                           Code.point("2")),
                                         Answer2, RefType::Explicit}}),
      UnorderedElementsAre(
          Pair(Code.point("1"), UnorderedElementsAre(HdrFile)),
          Pair(Code.point("2"), UnorderedElementsAre(HdrFile))));
}

TEST(Analyze, Basic) {
  TestInputs Inputs;
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

  RecordedPP PP;
  PragmaIncludes PI;
  Inputs.MakeAction = [&PP, &PI] {
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

  TestAST AST(Inputs);
  auto Decls = AST.context().getTranslationUnitDecl()->decls();
  auto Results =
      analyze(std::vector<Decl *>{Decls.begin(), Decls.end()},
              PP.MacroReferences, PP.Includes, &PI, AST.sourceManager(),
              AST.preprocessor().getHeaderSearchInfo());

  const Include *B = PP.Includes.atLine(3);
  ASSERT_EQ(B->Spelled, "b.h");
  EXPECT_THAT(Results.Missing, ElementsAre("\"c.h\""));
  EXPECT_THAT(Results.Unused, ElementsAre(B));
}

TEST(FixIncludes, Basic) {
  llvm::StringRef Code = R"cpp(
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

  EXPECT_EQ(fixIncludes(Results, Code, format::getLLVMStyle()), R"cpp(
#include "a.h"
#include "aa.h"
#include "ab.h"
#include <e.h>
)cpp");
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
             /*PragmaIncludes=*/nullptr, SM,
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
          << T.Main;
    } else {
      EXPECT_THAT(Main.points(), testing::IsEmpty());
    }
  }
}

TEST(Hints, Ordering) {
  struct Tag {};
  auto Hinted = [](Hints Hints) {
    return clang::include_cleaner::Hinted<Tag>({}, Hints);
  };
  EXPECT_LT(Hinted(Hints::None), Hinted(Hints::CompleteSymbol));
  EXPECT_LT(Hinted(Hints::CompleteSymbol), Hinted(Hints::PublicHeader));
  EXPECT_LT(Hinted(Hints::PublicHeader), Hinted(Hints::PreferredHeader));
  EXPECT_LT(Hinted(Hints::CompleteSymbol | Hints::PublicHeader),
            Hinted(Hints::PreferredHeader));
}

} // namespace
} // namespace clang::include_cleaner
