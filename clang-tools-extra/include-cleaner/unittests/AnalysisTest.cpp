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
#include "llvm/Testing/Support/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>

namespace clang::include_cleaner {
namespace {
using testing::Pair;
using testing::UnorderedElementsAre;

TEST(WalkUsed, Basic) {
  // FIXME: Have a fixture for setting up tests.
  llvm::Annotations Code(R"cpp(
  #include "header.h"
  #include "private.h"

  void $bar^bar($private^Private) {
    $foo^foo();
    std::$vector^vector $vconstructor^v;
  }
  )cpp");
  TestInputs Inputs(Code.code());
  Inputs.ExtraFiles["header.h"] = R"cpp(
  void foo();
  namespace std { class vector {}; }
  )cpp";
  Inputs.ExtraFiles["private.h"] = R"cpp(
    // IWYU pragma: private, include "path/public.h"
    class Private {};
  )cpp";

  PragmaIncludes PI;
  Inputs.MakeAction = [&PI] {
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
  TestAST AST(Inputs);

  llvm::SmallVector<Decl *> TopLevelDecls;
  for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
    TopLevelDecls.emplace_back(D);
  }

  auto &SM = AST.sourceManager();
  llvm::DenseMap<size_t, std::vector<Header>> OffsetToProviders;
  walkUsed(TopLevelDecls, /*MacroRefs=*/{}, PI, SM,
           [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
             auto [FID, Offset] = SM.getDecomposedLoc(Ref.RefLocation);
             EXPECT_EQ(FID, SM.getMainFileID());
             OffsetToProviders.try_emplace(Offset, Providers.vec());
           });
  auto HeaderFile = Header(AST.fileManager().getFile("header.h").get());
  auto MainFile = Header(SM.getFileEntryForID(SM.getMainFileID()));
  auto VectorSTL = Header(tooling::stdlib::Header::named("<vector>").value());
  EXPECT_THAT(
      OffsetToProviders,
      UnorderedElementsAre(
          Pair(Code.point("bar"), UnorderedElementsAre(MainFile)),
          Pair(Code.point("private"),
               UnorderedElementsAre(Header("\"path/public.h\""))),
          Pair(Code.point("foo"), UnorderedElementsAre(HeaderFile)),
          Pair(Code.point("vector"), UnorderedElementsAre(VectorSTL)),
          Pair(Code.point("vconstructor"), UnorderedElementsAre(VectorSTL))));
}

TEST(WalkUsed, MacroRefs) {
  llvm::Annotations Hdr(R"cpp(
    #define ^ANSWER 42
  )cpp");
  llvm::Annotations Main(R"cpp(
    #include "hdr.h"
    int x = ^ANSWER;
  )cpp");

  SourceManagerForFile SMF("main.cpp", Main.code());
  auto &SM = SMF.get();
  const FileEntry *HdrFile =
      SM.getFileManager().getVirtualFile("hdr.h", Hdr.code().size(), 0);
  SM.overrideFileContents(HdrFile,
                          llvm::MemoryBuffer::getMemBuffer(Hdr.code().str()));
  FileID HdrID = SM.createFileID(HdrFile, SourceLocation(), SrcMgr::C_User);

  IdentifierTable Idents;
  Symbol Answer =
      Macro{&Idents.get("ANSWER"), SM.getComposedLoc(HdrID, Hdr.point())};
  llvm::DenseMap<size_t, std::vector<Header>> OffsetToProviders;
  PragmaIncludes PI;
  walkUsed(/*ASTRoots=*/{}, /*MacroRefs=*/
           {SymbolReference{SM.getComposedLoc(SM.getMainFileID(), Main.point()),
                            Answer, RefType::Explicit}},
           PI, SM,
           [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
             auto [FID, Offset] = SM.getDecomposedLoc(Ref.RefLocation);
             EXPECT_EQ(FID, SM.getMainFileID());
             OffsetToProviders.try_emplace(Offset, Providers.vec());
           });

  EXPECT_THAT(
      OffsetToProviders,
      UnorderedElementsAre(Pair(Main.point(), UnorderedElementsAre(HdrFile))));
}

TEST(FindIncludeHeaders, IWYU) {
  TestInputs Inputs;
  PragmaIncludes PI;
  Inputs.MakeAction = [&PI] {
    struct Hook : public PreprocessOnlyAction {
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

  Inputs.Code = R"cpp(
    #include "header1.h"
    #include "header2.h"
  )cpp";
  Inputs.ExtraFiles["header1.h"] = R"cpp(
    // IWYU pragma: private, include "path/public.h"
  )cpp";
  Inputs.ExtraFiles["header2.h"] = R"cpp(
    #include "detail1.h" // IWYU pragma: export

    // IWYU pragma: begin_exports
    #include "detail2.h"
    // IWYU pragma: end_exports

    #include "normal.h"
  )cpp";
  Inputs.ExtraFiles["normal.h"] = Inputs.ExtraFiles["detail1.h"] =
      Inputs.ExtraFiles["detail2.h"] = "";
  TestAST AST(Inputs);
  const auto &SM = AST.sourceManager();
  auto &FM = SM.getFileManager();
  // Returns the source location for the start of the file.
  auto SourceLocFromFile = [&](llvm::StringRef FileName) {
    return SM.translateFileLineCol(FM.getFile(FileName).get(),
                                   /*Line=*/1, /*Col=*/1);
  };

  EXPECT_THAT(findIncludeHeaders(SourceLocFromFile("header1.h"), SM, PI),
              UnorderedElementsAre(Header("\"path/public.h\"")));

  EXPECT_THAT(findIncludeHeaders(SourceLocFromFile("detail1.h"), SM, PI),
              UnorderedElementsAre(Header(FM.getFile("header2.h").get()),
                                   Header(FM.getFile("detail1.h").get())));
  EXPECT_THAT(findIncludeHeaders(SourceLocFromFile("detail2.h"), SM, PI),
              UnorderedElementsAre(Header(FM.getFile("header2.h").get()),
                                   Header(FM.getFile("detail2.h").get())));

  EXPECT_THAT(findIncludeHeaders(SourceLocFromFile("normal.h"), SM, PI),
              UnorderedElementsAre(Header(FM.getFile("normal.h").get())));
}

} // namespace
} // namespace clang::include_cleaner
