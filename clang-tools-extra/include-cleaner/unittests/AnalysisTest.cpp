//===--- AnalysisTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
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
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>

namespace clang::include_cleaner {
namespace {
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
  auto VectorSTL = Header(tooling::stdlib::Header::named("<vector>").value());
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
    int x = ^ANSWER;
  )cpp");
  llvm::Annotations Hdr(guard("#define ^ANSWER 42"));
  Inputs.Code = Code.code();
  Inputs.ExtraFiles["hdr.h"] = Hdr.code();
  TestAST AST(Inputs);
  auto &SM = AST.sourceManager();
  auto HdrFile = SM.getFileManager().getFile("hdr.h").get();
  auto HdrID = SM.translateFile(HdrFile);

  IdentifierTable Idents;
  Symbol Answer =
      Macro{&Idents.get("ANSWER"), SM.getComposedLoc(HdrID, Hdr.point())};
  EXPECT_THAT(
      offsetToProviders(
          AST, SM,
          {SymbolReference{SM.getComposedLoc(SM.getMainFileID(), Code.point()),
                           Answer, RefType::Explicit}}),
      UnorderedElementsAre(Pair(Code.point(), UnorderedElementsAre(HdrFile))));
}

TEST(Analyze, Basic) {
  TestInputs Inputs;
  Inputs.Code = R"cpp(
#include "a.h"
#include "b.h"

int x = a + c;
)cpp";
  Inputs.ExtraFiles["a.h"] = guard("int a;");
  Inputs.ExtraFiles["b.h"] = guard(R"cpp(
    #include "c.h"
    int b;
  )cpp");
  Inputs.ExtraFiles["c.h"] = guard("int c;");

  RecordedPP PP;
  Inputs.MakeAction = [&PP] {
    struct Hook : public SyntaxOnlyAction {
    public:
      Hook(RecordedPP &PP) : PP(PP) {}
      bool BeginSourceFileAction(clang::CompilerInstance &CI) override {
        CI.getPreprocessor().addPPCallbacks(PP.record(CI.getPreprocessor()));
        return true;
      }

      RecordedPP &PP;
    };
    return std::make_unique<Hook>(PP);
  };

  TestAST AST(Inputs);
  auto Decls = AST.context().getTranslationUnitDecl()->decls();
  auto Results =
      analyze(std::vector<Decl *>{Decls.begin(), Decls.end()},
              PP.MacroReferences, PP.Includes, /*PragmaIncludes=*/nullptr,
              AST.sourceManager(), AST.preprocessor().getHeaderSearchInfo());

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

} // namespace
} // namespace clang::include_cleaner
