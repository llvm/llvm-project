//===--- FindHeadersTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "TypesInternal.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Testing/TestAST.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <memory>

namespace clang::include_cleaner {
namespace {
using testing::ElementsAre;
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

  llvm::SmallVector<Hinted<Header>> findHeaders(llvm::StringRef FileName) {
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

TEST_F(FindHeadersTest, IWYUExportForStandardHeaders) {
  Inputs.Code = R"cpp(
    #include "exporter.h"
  )cpp";
  Inputs.ExtraFiles["exporter.h"] = guard(R"cpp(
    #include <string> // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["string"] = guard("");
  Inputs.ExtraArgs.push_back("-isystem.");
  buildAST();
  tooling::stdlib::Symbol StdString =
      *tooling::stdlib::Symbol::named("std::", "string");
  EXPECT_THAT(
      include_cleaner::findHeaders(StdString, AST->sourceManager(), &PI),
      UnorderedElementsAre(physicalHeader("exporter.h"), StdString.header()));
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

    auto Headers = clang::include_cleaner::findHeaders(
        Visitor.Out->getLocation(), AST->sourceManager(),
        /*PragmaIncludes=*/nullptr);
    EXPECT_THAT(Headers, UnorderedElementsAre(physicalHeader("declare.h")));
  }
}

MATCHER_P2(HintedHeader, Header, Hint, "") {
  return std::tie(arg.Hint, arg) == std::tie(Hint, Header);
}

TEST_F(FindHeadersTest, PublicHeaderHint) {
  Inputs.Code = R"cpp(
    #include "public.h"
  )cpp";
  Inputs.ExtraFiles["public.h"] = guard(R"cpp(
    #include "private.h"
    #include "private.inc"
  )cpp");
  Inputs.ExtraFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private
  )cpp");
  Inputs.ExtraFiles["private.inc"] = "";
  buildAST();
  // Non self-contained files and headers marked with IWYU private pragma
  // shouldn't have PublicHeader hint.
  EXPECT_THAT(
      findHeaders("private.inc"),
      UnorderedElementsAre(
          HintedHeader(physicalHeader("private.inc"), Hints::OriginHeader),
          HintedHeader(physicalHeader("public.h"), Hints::PublicHeader)));
  EXPECT_THAT(findHeaders("private.h"),
              UnorderedElementsAre(HintedHeader(physicalHeader("private.h"),
                                                Hints::OriginHeader)));
}

TEST_F(FindHeadersTest, PreferredHeaderHint) {
  Inputs.Code = R"cpp(
    #include "private.h"
  )cpp";
  Inputs.ExtraFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public.h"
  )cpp");
  buildAST();
  // Headers explicitly marked should've preferred signal.
  EXPECT_THAT(
      findHeaders("private.h"),
      UnorderedElementsAre(
          HintedHeader(physicalHeader("private.h"), Hints::OriginHeader),
          HintedHeader(Header("\"public.h\""),
                       Hints::PreferredHeader | Hints::PublicHeader)));
}

class HeadersForSymbolTest : public FindHeadersTest {
protected:
  llvm::SmallVector<Header> headersFor(llvm::StringRef Name) {
    struct Visitor : public RecursiveASTVisitor<Visitor> {
      const NamedDecl *Out = nullptr;
      llvm::StringRef Name;
      Visitor(llvm::StringRef Name) : Name(Name) {}
      bool VisitNamedDecl(const NamedDecl *ND) {
        if (auto *TD = ND->getDescribedTemplate())
          ND = TD;

        if (ND->getName() == Name) {
          EXPECT_TRUE(Out == nullptr || Out == ND->getCanonicalDecl())
              << "Found multiple matches for " << Name << ".";
          Out = cast<NamedDecl>(ND->getCanonicalDecl());
        }
        return true;
      }
    };
    Visitor V(Name);
    V.TraverseDecl(AST->context().getTranslationUnitDecl());
    if (!V.Out)
      ADD_FAILURE() << "Couldn't find any decls named " << Name << ".";
    assert(V.Out);
    return headersForSymbol(*V.Out, AST->sourceManager(), &PI);
  }
  llvm::SmallVector<Header> headersForFoo() { return headersFor("foo"); }
};

TEST_F(HeadersForSymbolTest, Deduplicates) {
  Inputs.Code = R"cpp(
    #include "foo.h"
  )cpp";
  Inputs.ExtraFiles["foo.h"] = guard(R"cpp(
    // IWYU pragma: private, include "foo.h"
    void foo();
    void foo();
  )cpp");
  buildAST();
  EXPECT_THAT(
      headersForFoo(),
      UnorderedElementsAre(physicalHeader("foo.h"),
                           // FIXME: de-duplicate across different kinds.
                           Header("\"foo.h\"")));
}

TEST_F(HeadersForSymbolTest, RankByName) {
  Inputs.Code = R"cpp(
    #include "fox.h"
    #include "bar.h"
  )cpp";
  Inputs.ExtraFiles["fox.h"] = guard(R"cpp(
    void foo();
  )cpp");
  Inputs.ExtraFiles["bar.h"] = guard(R"cpp(
    void foo();
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("bar.h"), physicalHeader("fox.h")));
}

TEST_F(HeadersForSymbolTest, Ranking) {
  // Sorting is done over (canonical, public, complete, origin)-tuple.
  Inputs.Code = R"cpp(
    #include "private.h"
    #include "public.h"
    #include "public_complete.h"
    #include "exporter.h"
  )cpp";
  Inputs.ExtraFiles["public.h"] = guard(R"cpp(
    struct foo;
  )cpp");
  Inputs.ExtraFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private, include "canonical.h"
    struct foo;
  )cpp");
  Inputs.ExtraFiles["exporter.h"] = guard(R"cpp(
  #include "private.h" // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["public_complete.h"] = guard("struct foo {};");
  buildAST();
  EXPECT_THAT(headersForFoo(), ElementsAre(Header("\"canonical.h\""),
                                           physicalHeader("public_complete.h"),
                                           physicalHeader("public.h"),
                                           physicalHeader("exporter.h"),
                                           physicalHeader("private.h")));
}

TEST_F(HeadersForSymbolTest, PreferPublicOverComplete) {
  Inputs.Code = R"cpp(
    #include "complete_private.h"
    #include "public.h"
  )cpp";
  Inputs.ExtraFiles["complete_private.h"] = guard(R"cpp(
    // IWYU pragma: private
    struct foo {};
  )cpp");
  Inputs.ExtraFiles["public.h"] = guard("struct foo;");
  buildAST();
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("public.h"),
                          physicalHeader("complete_private.h")));
}

TEST_F(HeadersForSymbolTest, PreferNameMatch) {
  Inputs.Code = R"cpp(
    #include "public_complete.h"
    #include "test/foo.fwd.h"
  )cpp";
  Inputs.ExtraFiles["public_complete.h"] = guard(R"cpp(
    struct foo {};
  )cpp");
  Inputs.ExtraFiles["test/foo.fwd.h"] = guard("struct foo;");
  buildAST();
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("test/foo.fwd.h"),
                          physicalHeader("public_complete.h")));
}

TEST_F(HeadersForSymbolTest, MainFile) {
  Inputs.Code = R"cpp(
    #include "public_complete.h"
    struct foo;
  )cpp";
  Inputs.ExtraFiles["public_complete.h"] = guard(R"cpp(
    struct foo {};
  )cpp");
  buildAST();
  auto &SM = AST->sourceManager();
  // FIXME: Symbols provided by main file should be treated specially.
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("public_complete.h"),
                          Header(SM.getFileEntryForID(SM.getMainFileID()))));
}

TEST_F(HeadersForSymbolTest, PreferExporterOfPrivate) {
  Inputs.Code = R"cpp(
    #include "private.h"
    #include "exporter.h"
  )cpp";
  Inputs.ExtraFiles["private.h"] = guard(R"cpp(
    // IWYU pragma: private
    struct foo {};
  )cpp");
  Inputs.ExtraFiles["exporter.h"] = guard(R"cpp(
    #include "private.h" // IWYU pragma: export
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(), ElementsAre(physicalHeader("exporter.h"),
                                           physicalHeader("private.h")));
}

TEST_F(HeadersForSymbolTest, ExporterIsDownRanked) {
  Inputs.Code = R"cpp(
    #include "exporter.h"
    #include "zoo.h"
  )cpp";
  // Deliberately named as zoo to make sure it doesn't get name-match boost and
  // also gets lexicographically bigger order than "exporter".
  Inputs.ExtraFiles["zoo.h"] = guard(R"cpp(
    struct foo {};
  )cpp");
  Inputs.ExtraFiles["exporter.h"] = guard(R"cpp(
    #include "zoo.h" // IWYU pragma: export
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(), ElementsAre(physicalHeader("zoo.h"),
                                           physicalHeader("exporter.h")));
}

TEST_F(HeadersForSymbolTest, PreferPublicOverNameMatchOnPrivate) {
  Inputs.Code = R"cpp(
    #include "foo.h"
  )cpp";
  Inputs.ExtraFiles["foo.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public.h"
    struct foo {};
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(), ElementsAre(Header(StringRef("\"public.h\"")),
                                           physicalHeader("foo.h")));
}

TEST_F(HeadersForSymbolTest, PublicOverPrivateWithoutUmbrella) {
  Inputs.Code = R"cpp(
    #include "bar.h"
    #include "foo.h"
  )cpp";
  Inputs.ExtraFiles["bar.h"] =
      guard(R"cpp(#include "foo.h" // IWYU pragma: export)cpp");
  Inputs.ExtraFiles["foo.h"] = guard(R"cpp(
    // IWYU pragma: private
    struct foo {};
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("bar.h"), physicalHeader("foo.h")));
}

TEST_F(HeadersForSymbolTest, IWYUTransitiveExport) {
  Inputs.Code = R"cpp(
    #include "export1.h"
  )cpp";
  Inputs.ExtraFiles["export1.h"] = guard(R"cpp(
    #include "export2.h" // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["export2.h"] = guard(R"cpp(
    #include "foo.h" // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["foo.h"] = guard(R"cpp(
    struct foo {};
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("foo.h"), physicalHeader("export1.h"),
                          physicalHeader("export2.h")));
}

TEST_F(HeadersForSymbolTest, IWYUTransitiveExportWithPrivate) {
  Inputs.Code = R"cpp(
    #include "export1.h"
    void bar() { foo();}
  )cpp";
  Inputs.ExtraFiles["export1.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public1.h"
    #include "export2.h" // IWYU pragma: export
    void foo();
  )cpp");
  Inputs.ExtraFiles["export2.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public2.h"
    #include "export3.h" // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["export3.h"] = guard(R"cpp(
    // IWYU pragma: private, include "public3.h"
    #include "foo.h" // IWYU pragma: export
  )cpp");
  Inputs.ExtraFiles["foo.h"] = guard(R"cpp(
    void foo();
  )cpp");
  buildAST();
  EXPECT_THAT(headersForFoo(),
              ElementsAre(physicalHeader("foo.h"),
                                           Header(StringRef("\"public1.h\"")),
                                           physicalHeader("export1.h"),
                                           physicalHeader("export2.h"),
                                           physicalHeader("export3.h")));
}

TEST_F(HeadersForSymbolTest, AmbiguousStdSymbols) {
  struct {
    llvm::StringRef Code;
    llvm::StringRef Name;

    llvm::StringRef ExpectedHeader;
  } TestCases[] = {
      {
          R"cpp(
            namespace std {
             template <typename InputIt, typename OutputIt>
             constexpr OutputIt move(InputIt first, InputIt last, OutputIt dest);
            })cpp",
          "move",
          "<algorithm>",
      },
      {
          R"cpp(
            namespace std {
             template<class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
             ForwardIt2 move(ExecutionPolicy&& policy,
                 ForwardIt1 first, ForwardIt1 last, ForwardIt2 d_first);
            })cpp",
          "move",
          "<algorithm>",
      },
      {
          R"cpp(
            namespace std {
              template<typename T> constexpr T move(T&& t) noexcept;
            })cpp",
          "move",
          "<utility>",
      },
      {
          R"cpp(
            namespace std {
              template<class ForwardIt, class T>
              ForwardIt remove(ForwardIt first, ForwardIt last, const T& value);
            })cpp",
          "remove",
          "<algorithm>",
      },
      {
          "namespace std { int remove(const char*); }",
          "remove",
          "<cstdio>",
      },
  };

  for (const auto &T : TestCases) {
    Inputs.Code = T.Code;
    buildAST();
    EXPECT_THAT(headersFor(T.Name),
                UnorderedElementsAre(
                    Header(*tooling::stdlib::Header::named(T.ExpectedHeader))));
  }
}

TEST_F(HeadersForSymbolTest, StandardHeaders) {
  Inputs.Code = "void assert();";
  buildAST();
  EXPECT_THAT(
      headersFor("assert"),
      // Respect the ordering from the stdlib mapping.
      UnorderedElementsAre(tooling::stdlib::Header::named("<cassert>"),
                           tooling::stdlib::Header::named("<assert.h>")));
}

} // namespace
} // namespace clang::include_cleaner
