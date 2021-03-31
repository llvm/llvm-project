//===- unittest/Tooling/CrossTranslationUnitTest.cpp - Tooling unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CrossTU/CrossTranslationUnit.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Optional.h"
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

  void HandleTranslationUnit(ASTContext &Ctx) override {
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

    const DynTypedNodeList ParentsBeforeImport =
        Ctx.getParentMapContext().getParents<Decl>(*FD);
    ASSERT_FALSE(ParentsBeforeImport.empty());

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
        // Check parent map.
        const DynTypedNodeList ParentsAfterImport =
            Ctx.getParentMapContext().getParents<Decl>(*FD);
        const DynTypedNodeList ParentsOfImported =
            Ctx.getParentMapContext().getParents<Decl>(*NewFD);
        EXPECT_TRUE(
            checkParentListsEq(ParentsBeforeImport, ParentsAfterImport));
        EXPECT_FALSE(ParentsOfImported.empty());
      }
    }
  }

  static bool checkParentListsEq(const DynTypedNodeList &L1,
                                 const DynTypedNodeList &L2) {
    if (L1.size() != L2.size())
      return false;
    for (unsigned int I = 0; I < L1.size(); ++I)
      if (L1[I] != L2[I])
        return false;
    return true;
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
    CI.getAnalyzerOpts()->CTUImportCppThreshold = OverrideLimit;
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
      parseCrossTUIndex(IndexFileName);
  EXPECT_TRUE((bool)IndexOrErr);
  llvm::StringMap<std::string> ParsedIndex = IndexOrErr.get();
  for (const auto &E : Index) {
    EXPECT_TRUE(ParsedIndex.count(E.getKey()));
    EXPECT_EQ(ParsedIndex[E.getKey()], E.getValue());
  }
  for (const auto &E : ParsedIndex)
    EXPECT_TRUE(Index.count(E.getKey()));
}

TEST(CrossTranslationUnit, EmptyInvocationListIsNotValid) {
  auto Input = "";

  llvm::Expected<InvocationListTy> Result = parseInvocationList(Input);
  EXPECT_FALSE(static_cast<bool>(Result));
  bool IsWrongFromatError = false;
  llvm::handleAllErrors(Result.takeError(), [&](IndexError &Err) {
    IsWrongFromatError =
        Err.getCode() == index_error_code::invocation_list_wrong_format;
  });
  EXPECT_TRUE(IsWrongFromatError);
}

TEST(CrossTranslationUnit, AmbiguousInvocationListIsDetected) {
  // The same source file occurs twice (for two different architecture) in
  // this test case. The disambiguation is the responsibility of the user.
  auto Input = R"(
  /tmp/main.cpp:
    - clang++
    - -c
    - -m32
    - -o
    - main32.o
    - /tmp/main.cpp
  /tmp/main.cpp:
    - clang++
    - -c
    - -m64
    - -o
    - main64.o
    - /tmp/main.cpp
  )";

  llvm::Expected<InvocationListTy> Result = parseInvocationList(Input);
  EXPECT_FALSE(static_cast<bool>(Result));
  bool IsAmbiguousError = false;
  llvm::handleAllErrors(Result.takeError(), [&](IndexError &Err) {
    IsAmbiguousError =
        Err.getCode() == index_error_code::invocation_list_ambiguous;
  });
  EXPECT_TRUE(IsAmbiguousError);
}

TEST(CrossTranslationUnit, SingleInvocationCanBeParsed) {
  auto Input = R"(
  /tmp/main.cpp:
    - clang++
    - /tmp/main.cpp
  )";
  llvm::Expected<InvocationListTy> Result = parseInvocationList(Input);
  EXPECT_TRUE(static_cast<bool>(Result));

  EXPECT_EQ(Result->size(), 1u);

  auto It = Result->find("/tmp/main.cpp");
  EXPECT_TRUE(It != Result->end());
  EXPECT_EQ(It->getValue()[0], "clang++");
  EXPECT_EQ(It->getValue()[1], "/tmp/main.cpp");
}

TEST(CrossTranslationUnit, MultipleInvocationsCanBeParsed) {
  auto Input = R"(
  /tmp/main.cpp:
    - clang++
    - /tmp/other.o
    - /tmp/main.cpp
  /tmp/other.cpp:
    - g++
    - -c
    - -o
    - /tmp/other.o
    - /tmp/other.cpp
  )";
  llvm::Expected<InvocationListTy> Result = parseInvocationList(Input);
  EXPECT_TRUE(static_cast<bool>(Result));

  EXPECT_EQ(Result->size(), 2u);

  auto It = Result->find("/tmp/main.cpp");
  EXPECT_TRUE(It != Result->end());
  EXPECT_EQ(It->getKey(), "/tmp/main.cpp");
  EXPECT_EQ(It->getValue()[0], "clang++");
  EXPECT_EQ(It->getValue()[1], "/tmp/other.o");
  EXPECT_EQ(It->getValue()[2], "/tmp/main.cpp");

  It = Result->find("/tmp/other.cpp");
  EXPECT_TRUE(It != Result->end());
  EXPECT_EQ(It->getValue()[0], "g++");
  EXPECT_EQ(It->getValue()[1], "-c");
  EXPECT_EQ(It->getValue()[2], "-o");
  EXPECT_EQ(It->getValue()[3], "/tmp/other.o");
  EXPECT_EQ(It->getValue()[4], "/tmp/other.cpp");
}

} // end namespace cross_tu
} // end namespace clang
