//===--- FindHeadersTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/FileManager.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Testing/TestAST.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang::include_cleaner {
namespace {
using testing::UnorderedElementsAre;

std::string guard(llvm::StringRef Code) {
  return "#pragma once\n" + Code.str();
}

class FindHeadersTest : public testing::Test {
protected:
  TestInputs Inputs;
  PragmaIncludes PI;
  std::unique_ptr<TestAST> AST;
  FindHeadersTest() {
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
  void buildAST() { AST = std::make_unique<TestAST>(Inputs); }

  llvm::SmallVector<Header> findHeaders(llvm::StringRef FileName) {
    return include_cleaner::findHeaders(
        AST->sourceManager().translateFileLineCol(
            AST->fileManager().getFile(FileName).get(),
            /*Line=*/1, /*Col=*/1),
        AST->sourceManager(), &PI);
  }
  const FileEntry *physicalHeader(llvm::StringRef FileName) {
    return AST->fileManager().getFile(FileName).get();
  };
};

TEST_F(FindHeadersTest, IWYUPrivateToPublic) {
  Inputs.Code = R"cpp(
    #include "private.h"
  )cpp";
  Inputs.ExtraFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "path/public.h"
  )cpp");
  buildAST();
  EXPECT_THAT(findHeaders("private.h"),
              UnorderedElementsAre(physicalHeader("private.h"),
                                   Header("\"path/public.h\"")));
}

TEST_F(FindHeadersTest, IWYUExport) {
  Inputs.Code = R"cpp(
    #include "exporter.h"
  )cpp";
  Inputs.ExtraFiles["exporter.h"] = guard(R"cpp(
    #include "exported1.h" // IWYU pragma: export

    // IWYU pragma: begin_exports
    #include "exported2.h"
    // IWYU pragma: end_exports

    #include "normal.h"
  )cpp");
  Inputs.ExtraFiles["exported1.h"] = guard("");
  Inputs.ExtraFiles["exported2.h"] = guard("");
  Inputs.ExtraFiles["normal.h"] = guard("");

  buildAST();
  EXPECT_THAT(findHeaders("exported1.h"),
              UnorderedElementsAre(physicalHeader("exported1.h"),
                                   physicalHeader("exporter.h")));
  EXPECT_THAT(findHeaders("exported2.h"),
              UnorderedElementsAre(physicalHeader("exported2.h"),
                                   physicalHeader("exporter.h")));
  EXPECT_THAT(findHeaders("normal.h"),
              UnorderedElementsAre(physicalHeader("normal.h")));
  EXPECT_THAT(findHeaders("exporter.h"),
              UnorderedElementsAre(physicalHeader("exporter.h")));
}

TEST_F(FindHeadersTest, SelfContained) {
  Inputs.Code = R"cpp(
    #include "header.h"
  )cpp";
  Inputs.ExtraFiles["header.h"] = guard(R"cpp(
    #include "fragment.inc"
  )cpp");
  Inputs.ExtraFiles["fragment.inc"] = "";
  buildAST();
  EXPECT_THAT(findHeaders("fragment.inc"),
              UnorderedElementsAre(physicalHeader("fragment.inc"),
                                   physicalHeader("header.h")));
}

TEST_F(FindHeadersTest, NonSelfContainedTraversePrivate) {
  Inputs.Code = R"cpp(
    #include "header.h"
  )cpp";
  Inputs.ExtraFiles["header.h"] = guard(R"cpp(
    #include "fragment.inc"
  )cpp");
  Inputs.ExtraFiles["fragment.inc"] = R"cpp(
    // IWYU pragma: private, include "public.h"
  )cpp";

  buildAST();
  // There is a IWYU private mapping in the non self-contained header, verify
  // that we don't emit its includer.
  EXPECT_THAT(findHeaders("fragment.inc"),
              UnorderedElementsAre(physicalHeader("fragment.inc"),
                                   Header("\"public.h\"")));
}

TEST_F(FindHeadersTest, NonSelfContainedTraverseExporter) {
  Inputs.Code = R"cpp(
    #include "exporter.h"
  )cpp";
  Inputs.ExtraFiles["exporter.h"] = guard(R"cpp(
    #include "exported.h" // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["exported.h"] = guard(R"cpp(
    #include "fragment.inc"
  )cpp");
  Inputs.ExtraFiles["fragment.inc"] = "";
  buildAST();
  // Verify that we emit exporters for each header on the path.
  EXPECT_THAT(findHeaders("fragment.inc"),
              UnorderedElementsAre(physicalHeader("fragment.inc"),
                                   physicalHeader("exported.h"),
                                   physicalHeader("exporter.h")));
}

TEST_F(FindHeadersTest, TargetIsExpandedFromMacroInHeader) {
  struct CustomVisitor : RecursiveASTVisitor<CustomVisitor> {
    const Decl *Out = nullptr;
    bool VisitNamedDecl(const NamedDecl *ND) {
      if (ND->getName() == "FLAG_foo" || ND->getName() == "Foo") {
        EXPECT_TRUE(Out == nullptr);
        Out = ND;
      }
      return true;
    }
  };

  struct {
    llvm::StringRef MacroHeader;
    llvm::StringRef DeclareHeader;
  } TestCases[] = {
      {/*MacroHeader=*/R"cpp(
    #define DEFINE_CLASS(name) class name {};
  )cpp",
       /*DeclareHeader=*/R"cpp(
    #include "macro.h"
    DEFINE_CLASS(Foo)
  )cpp"},
      {/*MacroHeader=*/R"cpp(
    #define DEFINE_Foo class Foo {};
  )cpp",
       /*DeclareHeader=*/R"cpp(
    #include "macro.h"
    DEFINE_Foo
  )cpp"},
      {/*MacroHeader=*/R"cpp(
    #define DECLARE_FLAGS(name) extern int FLAG_##name
  )cpp",
       /*DeclareHeader=*/R"cpp(
    #include "macro.h"
    DECLARE_FLAGS(foo);
  )cpp"},
  };

  for (const auto &T : TestCases) {
    Inputs.Code = R"cpp(#include "declare.h")cpp";
    Inputs.ExtraFiles["declare.h"] = guard(T.DeclareHeader);
    Inputs.ExtraFiles["macro.h"] = guard(T.MacroHeader);
    buildAST();

    CustomVisitor Visitor;
    Visitor.TraverseDecl(AST->context().getTranslationUnitDecl());

    llvm::SmallVector<Header> Headers = clang::include_cleaner::findHeaders(
        Visitor.Out->getLocation(), AST->sourceManager(),
        /*PragmaIncludes=*/nullptr);
    EXPECT_THAT(Headers, UnorderedElementsAre(physicalHeader("declare.h")));
  }
}

} // namespace
} // namespace clang::include_cleaner
