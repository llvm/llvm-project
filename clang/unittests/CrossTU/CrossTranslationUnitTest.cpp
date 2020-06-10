//===- unittest/Tooling/CrossTranslationUnitTest.cpp - Tooling unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "gtest/gtest.h"
#include <cassert>

namespace clang {
namespace cross_tu {

namespace {

class CTUASTConsumer : public clang::ASTConsumer {
public:
  explicit CTUASTConsumer(clang::CompilerInstance &CI, bool *Success)
      : CTU(CI), Success(Success) {}

  void HandleTranslationUnit(ASTContext &Ctx) {
    auto FindFInTU = [](const TranslationUnitDecl *TU) {
      const FunctionDecl *FD = nullptr;
      for (const Decl *D : TU->decls()) {
        FD = dyn_cast<FunctionDecl>(D);
        if (FD && FD->getName() == "f")
          break;
      }
      return FD;
    };

    const TranslationUnitDecl *TU = Ctx.getTranslationUnitDecl();
    const FunctionDecl *FD = FindFInTU(TU);
    assert(FD && FD->getName() == "f");
    bool OrigFDHasBody = FD->hasBody();

    // Prepare the index file and the AST file.
    int ASTFD;
    llvm::SmallString<256> ASTFileName;
    ASSERT_FALSE(
        llvm::sys::fs::createTemporaryFile("f_ast", "ast", ASTFD, ASTFileName));
    llvm::ToolOutputFile ASTFile(ASTFileName, ASTFD);

    int IndexFD;
    llvm::SmallString<256> IndexFileName;
    ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("index", "txt", IndexFD,
                                                    IndexFileName));
    llvm::ToolOutputFile IndexFile(IndexFileName, IndexFD);
    IndexFile.os() << "c:@F@f#I# " << ASTFileName << "\n";
    IndexFile.os().flush();
    EXPECT_TRUE(llvm::sys::fs::exists(IndexFileName));

    StringRef SourceText = "int f(int) { return 0; }\n";
    // This file must exist since the saved ASTFile will reference it.
    int SourceFD;
    llvm::SmallString<256> SourceFileName;
    ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("input", "cpp", SourceFD,
                                                    SourceFileName));
    llvm::ToolOutputFile SourceFile(SourceFileName, SourceFD);
    SourceFile.os() << SourceText;
    SourceFile.os().flush();
    EXPECT_TRUE(llvm::sys::fs::exists(SourceFileName));

    std::unique_ptr<ASTUnit> ASTWithDefinition =
        tooling::buildASTFromCode(SourceText, SourceFileName);
    ASTWithDefinition->Save(ASTFileName.str());
    EXPECT_TRUE(llvm::sys::fs::exists(ASTFileName));

    // Load the definition from the AST file.
    llvm::Expected<const FunctionDecl *> NewFDorError = handleExpected(
        CTU.getCrossTUDefinition(FD, "", IndexFileName, false),
        []() { return nullptr; }, [](IndexError &) {});

    if (NewFDorError) {
      const FunctionDecl *NewFD = *NewFDorError;
      *Success = NewFD && NewFD->hasBody() && !OrigFDHasBody;

      if (NewFD) {
        // Check GetImportedFromSourceLocation.
        llvm::Optional<std::pair<SourceLocation, ASTUnit *>> SLocResult =
            CTU.getImportedFromSourceLocation(NewFD->getLocation());
        EXPECT_TRUE(SLocResult);
        if (SLocResult) {
          SourceLocation OrigSLoc = (*SLocResult).first;
          ASTUnit *OrigUnit = (*SLocResult).second;
          // OrigUnit is created internally by CTU (is not the
          // ASTWithDefinition).
          TranslationUnitDecl *OrigTU =
              OrigUnit->getASTContext().getTranslationUnitDecl();
          const FunctionDecl *FDWithDefinition = FindFInTU(OrigTU);
          EXPECT_TRUE(FDWithDefinition);
          if (FDWithDefinition) {
            EXPECT_EQ(FDWithDefinition->getName(), "f");
            EXPECT_TRUE(FDWithDefinition->isThisDeclarationADefinition());
            EXPECT_EQ(OrigSLoc, FDWithDefinition->getLocation());
          }
        }
      }
    }
  }

private:
  CrossTranslationUnitContext CTU;
  bool *Success;
};

class CTUAction : public clang::ASTFrontendAction {
public:
  CTUAction(bool *Success, unsigned OverrideLimit)
      : Success(Success), OverrideLimit(OverrideLimit) {}

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, StringRef) override {
    CI.getAnalyzerOpts()->CTUImportThreshold = OverrideLimit;
    return std::make_unique<CTUASTConsumer>(CI, Success);
  }

private:
  bool *Success;
  const unsigned OverrideLimit;
};

} // end namespace

TEST(CrossTranslationUnit, CanLoadFunctionDefinition) {
  bool Success = false;
  EXPECT_TRUE(tooling::runToolOnCode(std::make_unique<CTUAction>(&Success, 1u),
                                     "int f(int);"));
  EXPECT_TRUE(Success);
}

TEST(CrossTranslationUnit, RespectsLoadThreshold) {
  bool Success = false;
  EXPECT_TRUE(tooling::runToolOnCode(std::make_unique<CTUAction>(&Success, 0u),
                                     "int f(int);"));
  EXPECT_FALSE(Success);
}

TEST(CrossTranslationUnit, IndexFormatCanBeParsed) {
  llvm::StringMap<std::string> Index;
  Index["a"] = "/b/f1";
  Index["c"] = "/d/f2";
  Index["e"] = "/f/f3";
  std::string IndexText = createCrossTUIndexString(Index);

  int IndexFD;
  llvm::SmallString<256> IndexFileName;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("index", "txt", IndexFD,
                                                  IndexFileName));
  llvm::ToolOutputFile IndexFile(IndexFileName, IndexFD);
  IndexFile.os() << IndexText;
  IndexFile.os().flush();
  EXPECT_TRUE(llvm::sys::fs::exists(IndexFileName));
  llvm::Expected<llvm::StringMap<std::string>> IndexOrErr =
      parseCrossTUIndex(IndexFileName, "");
  EXPECT_TRUE((bool)IndexOrErr);
  llvm::StringMap<std::string> ParsedIndex = IndexOrErr.get();
  for (const auto &E : Index) {
    EXPECT_TRUE(ParsedIndex.count(E.getKey()));
    EXPECT_EQ(ParsedIndex[E.getKey()], E.getValue());
  }
  for (const auto &E : ParsedIndex)
    EXPECT_TRUE(Index.count(E.getKey()));
}

TEST(CrossTranslationUnit, CTUDirIsHandledCorrectly) {
  llvm::StringMap<std::string> Index;
  Index["a"] = "/b/c/d";
  std::string IndexText = createCrossTUIndexString(Index);

  int IndexFD;
  llvm::SmallString<256> IndexFileName;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("index", "txt", IndexFD,
                                                  IndexFileName));
  llvm::ToolOutputFile IndexFile(IndexFileName, IndexFD);
  IndexFile.os() << IndexText;
  IndexFile.os().flush();
  EXPECT_TRUE(llvm::sys::fs::exists(IndexFileName));
  llvm::Expected<llvm::StringMap<std::string>> IndexOrErr =
      parseCrossTUIndex(IndexFileName, "/ctudir");
  EXPECT_TRUE((bool)IndexOrErr);
  llvm::StringMap<std::string> ParsedIndex = IndexOrErr.get();
  EXPECT_EQ(ParsedIndex["a"], "/ctudir/b/c/d");
}

} // end namespace cross_tu
} // end namespace clang
