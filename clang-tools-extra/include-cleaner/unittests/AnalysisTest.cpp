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
  walkUsed(TopLevelDecls, [&](SourceLocation RefLoc, Symbol S,
                              llvm::ArrayRef<Header> Providers) {
    auto [FID, Offset] = SM.getDecomposedLoc(RefLoc);
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

} // namespace
} // namespace clang::include_cleaner
