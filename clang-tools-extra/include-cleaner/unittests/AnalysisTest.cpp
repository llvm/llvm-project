//===--- AnalysisTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
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
  llvm::Annotations HeaderCode(R"cpp(
  void foo();
  namespace std { class vector {}; })cpp");
  llvm::Annotations Code(R"cpp(
  void $bar^bar() {
    $foo^foo();
    std::$vector^vector $vconstructor^v;
  }
  )cpp");
  TestInputs Inputs(Code.code());
  Inputs.ExtraFiles["header.h"] = HeaderCode.code().str();
  Inputs.ExtraArgs.push_back("-include");
  Inputs.ExtraArgs.push_back("header.h");
  TestAST AST(Inputs);

  llvm::SmallVector<Decl *> TopLevelDecls;
  for (Decl *D : AST.context().getTranslationUnitDecl()->decls()) {
    TopLevelDecls.emplace_back(D);
  }

  auto &SM = AST.sourceManager();
  llvm::DenseMap<size_t, std::vector<Header>> OffsetToProviders;
  walkUsed(TopLevelDecls, /*MacroRefs=*/{}, SM,
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
  walkUsed(/*ASTRoots=*/{}, /*MacroRefs=*/
           {SymbolReference{SM.getComposedLoc(SM.getMainFileID(), Main.point()),
                            Answer, RefType::Explicit}},
           SM,
           [&](const SymbolReference &Ref, llvm::ArrayRef<Header> Providers) {
             auto [FID, Offset] = SM.getDecomposedLoc(Ref.RefLocation);
             EXPECT_EQ(FID, SM.getMainFileID());
             OffsetToProviders.try_emplace(Offset, Providers.vec());
           });

  EXPECT_THAT(
      OffsetToProviders,
      UnorderedElementsAre(Pair(Main.point(), UnorderedElementsAre(HdrFile))));
}

} // namespace
} // namespace clang::include_cleaner
