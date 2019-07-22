//==- SemanticHighlightingTests.cpp - SemanticHighlighting tests-*- C++ -* -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "Protocol.h"
#include "SemanticHighlighting.h"
#include "TestFS.h"
#include "TestTU.h"
#include "gmock/gmock.h"

namespace clang {
namespace clangd {
namespace {

std::vector<HighlightingToken>
makeHighlightingTokens(llvm::ArrayRef<Range> Ranges, HighlightingKind Kind) {
  std::vector<HighlightingToken> Tokens(Ranges.size());
  for (int I = 0, End = Ranges.size(); I < End; ++I) {
    Tokens[I].R = Ranges[I];
    Tokens[I].Kind = Kind;
  }

  return Tokens;
}

void checkHighlightings(llvm::StringRef Code) {
  Annotations Test(Code);
  auto AST = TestTU::withCode(Test.code()).build();
  static const std::map<HighlightingKind, std::string> KindToString{
      {HighlightingKind::Variable, "Variable"},
      {HighlightingKind::Function, "Function"},
      {HighlightingKind::Class, "Class"},
      {HighlightingKind::Enum, "Enum"},
      {HighlightingKind::Namespace, "Namespace"},
      {HighlightingKind::EnumConstant, "EnumConstant"},
      {HighlightingKind::Field, "Field"},
      {HighlightingKind::Method, "Method"},
      {HighlightingKind::TemplateParameter, "TemplateParameter"}};
  std::vector<HighlightingToken> ExpectedTokens;
  for (const auto &KindString : KindToString) {
    std::vector<HighlightingToken> Toks = makeHighlightingTokens(
        Test.ranges(KindString.second), KindString.first);
    ExpectedTokens.insert(ExpectedTokens.end(), Toks.begin(), Toks.end());
  }

  auto ActualTokens = getSemanticHighlightings(AST);
  EXPECT_THAT(ActualTokens, testing::UnorderedElementsAreArray(ExpectedTokens))
   << "Inputs is:\n" << Code;
}

TEST(SemanticHighlighting, GetsCorrectTokens) {
  const char *TestCases[] = {
    R"cpp(
      struct $Class[[AS]] {
        double $Field[[SomeMember]];
      };
      struct {
      } $Variable[[S]];
      void $Function[[foo]](int $Variable[[A]], $Class[[AS]] $Variable[[As]]) {
        auto $Variable[[VeryLongVariableName]] = 12312;
        $Class[[AS]]     $Variable[[AA]];
        auto $Variable[[L]] = $Variable[[AA]].$Field[[SomeMember]] + $Variable[[A]];
        auto $Variable[[FN]] = [ $Variable[[AA]]](int $Variable[[A]]) -> void {};
        $Variable[[FN]](12312);
      }
    )cpp",
    R"cpp(
      void $Function[[foo]](int);
      void $Function[[Gah]]();
      void $Function[[foo]]() {
        auto $Variable[[Bou]] = $Function[[Gah]];
      }
      struct $Class[[A]] {
        void $Method[[abc]]();
      };
    )cpp",
    R"cpp(
      namespace $Namespace[[abc]] {
        template<typename $TemplateParameter[[T]]>
        struct $Class[[A]] {
          $TemplateParameter[[T]] $Field[[t]];
        };
      }
      template<typename $TemplateParameter[[T]]>
      struct $Class[[C]] : $Namespace[[abc]]::$Class[[A]]<$TemplateParameter[[T]]> {
        typename $TemplateParameter[[T]]::A* $Field[[D]];
      };
      $Namespace[[abc]]::$Class[[A]]<int> $Variable[[AA]];
      typedef $Namespace[[abc]]::$Class[[A]]<int> $Class[[AAA]];
      struct $Class[[B]] {
        $Class[[B]]();
        ~$Class[[B]]();
        void operator<<($Class[[B]]);
        $Class[[AAA]] $Field[[AA]];
      };
      $Class[[B]]::$Class[[B]]() {}
      $Class[[B]]::~$Class[[B]]() {}
      void $Function[[f]] () {
        $Class[[B]] $Variable[[BB]] = $Class[[B]]();
        $Variable[[BB]].~$Class[[B]]();
        $Class[[B]]();
      }
    )cpp",
    R"cpp(
      enum class $Enum[[E]] {
        $EnumConstant[[A]],
        $EnumConstant[[B]],
      };
      enum $Enum[[EE]] {
        $EnumConstant[[Hi]],
      };
      struct $Class[[A]] {
        $Enum[[E]] $Field[[EEE]];
        $Enum[[EE]] $Field[[EEEE]];
      };
      int $Variable[[I]] = $EnumConstant[[Hi]];
      $Enum[[E]] $Variable[[L]] = $Enum[[E]]::$EnumConstant[[B]];
    )cpp",
    R"cpp(
      namespace $Namespace[[abc]] {
        namespace {}
        namespace $Namespace[[bcd]] {
          struct $Class[[A]] {};
          namespace $Namespace[[cde]] {
            struct $Class[[A]] {
              enum class $Enum[[B]] {
                $EnumConstant[[Hi]],
              };
            };
          }
        }
      }
      using namespace $Namespace[[abc]]::$Namespace[[bcd]];
      namespace $Namespace[[vwz]] =
            $Namespace[[abc]]::$Namespace[[bcd]]::$Namespace[[cde]];
      $Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable[[AA]];
      $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]] $Variable[[AAA]] =
            $Namespace[[vwz]]::$Class[[A]]::$Enum[[B]]::$EnumConstant[[Hi]];
      ::$Namespace[[vwz]]::$Class[[A]] $Variable[[B]];
      ::$Namespace[[abc]]::$Namespace[[bcd]]::$Class[[A]] $Variable[[BB]];
    )cpp",
    R"cpp(
      struct $Class[[D]] {
        double $Field[[C]];
      };
      struct $Class[[A]] {
        double $Field[[B]];
        $Class[[D]] $Field[[E]];
        static double $Variable[[S]];
        void $Method[[foo]]() {
          $Field[[B]] = 123;
          this->$Field[[B]] = 156;
          this->$Method[[foo]]();
          $Method[[foo]]();
          $Variable[[S]] = 90.1;
        }
      };
      void $Function[[foo]]() {
        $Class[[A]] $Variable[[AA]];
        $Variable[[AA]].$Field[[B]] += 2;
        $Variable[[AA]].$Method[[foo]]();
        $Variable[[AA]].$Field[[E]].$Field[[C]];
        $Class[[A]]::$Variable[[S]] = 90;
      }
    )cpp",
    R"cpp(
      struct $Class[[AA]] {
        int $Field[[A]];
      }
      int $Variable[[B]];
      $Class[[AA]] $Variable[[A]]{$Variable[[B]]};
    )cpp",
    R"cpp(
      namespace $Namespace[[a]] {
        struct $Class[[A]] {};
      }
      typedef $Namespace[[a]]::$Class[[A]] $Class[[B]];
      using $Class[[BB]] = $Namespace[[a]]::$Class[[A]];
      enum class $Enum[[E]] {};
      typedef $Enum[[E]] $Enum[[C]];
      typedef $Enum[[C]] $Enum[[CC]];
      using $Enum[[CD]] = $Enum[[CC]];
      $Enum[[CC]] $Function[[f]]($Class[[B]]);
      $Enum[[CD]] $Function[[f]]($Class[[BB]]);
    )cpp",
    R"cpp(
      template<typename $TemplateParameter[[T]], typename = void>
      class $Class[[A]] {
        $TemplateParameter[[T]] $Field[[AA]];
        $TemplateParameter[[T]] $Method[[foo]]();
      };
      template<class $TemplateParameter[[TT]]>
      class $Class[[B]] {
        $Class[[A]]<$TemplateParameter[[TT]]> $Field[[AA]];
      };
      template<class $TemplateParameter[[TT]], class $TemplateParameter[[GG]]>
      class $Class[[BB]] {};
      template<class $TemplateParameter[[T]]>
      class $Class[[BB]]<$TemplateParameter[[T]], int> {};
      template<class $TemplateParameter[[T]]>
      class $Class[[BB]]<$TemplateParameter[[T]], $TemplateParameter[[T]]*> {};

      template<template<class> class $TemplateParameter[[T]], class $TemplateParameter[[C]]>
      $TemplateParameter[[T]]<$TemplateParameter[[C]]> $Function[[f]]();

      template<typename>
      class $Class[[Foo]] {};

      template<typename $TemplateParameter[[T]]>
      void $Function[[foo]]($TemplateParameter[[T]] ...);
    )cpp"};
  for (const auto &TestCase : TestCases) {
    checkHighlightings(TestCase);
  }
}

TEST(SemanticHighlighting, GeneratesHighlightsWhenFileChange) {
  class HighlightingsCounterDiagConsumer : public DiagnosticsConsumer {
  public:
    std::atomic<int> Count = {0};

    void onDiagnosticsReady(PathRef, std::vector<Diag>) override {}
    void onHighlightingsReady(
        PathRef File, std::vector<HighlightingToken> Highlightings) override {
      ++Count;
    }
  };

  auto FooCpp = testPath("foo.cpp");
  MockFSProvider FS;
  FS.Files[FooCpp] = "";

  MockCompilationDatabase MCD;
  HighlightingsCounterDiagConsumer DiagConsumer;
  ClangdServer Server(MCD, FS, DiagConsumer, ClangdServer::optsForTest());
  Server.addDocument(FooCpp, "int a;");
  ASSERT_TRUE(Server.blockUntilIdleForTest()) << "Waiting for server";
  ASSERT_EQ(DiagConsumer.Count, 1);
}

TEST(SemanticHighlighting, toSemanticHighlightingInformation) {
  auto CreatePosition = [](int Line, int Character) -> Position {
    Position Pos;
    Pos.line = Line;
    Pos.character = Character;
    return Pos;
  };

  std::vector<HighlightingToken> Tokens{
      {HighlightingKind::Variable,
                        Range{CreatePosition(3, 8), CreatePosition(3, 12)}},
      {HighlightingKind::Function,
                        Range{CreatePosition(3, 4), CreatePosition(3, 7)}},
      {HighlightingKind::Variable,
                        Range{CreatePosition(1, 1), CreatePosition(1, 5)}}};
  std::vector<SemanticHighlightingInformation> ActualResults =
      toSemanticHighlightingInformation(Tokens);
  std::vector<SemanticHighlightingInformation> ExpectedResults = {
      {1, "AAAAAQAEAAA="},
      {3, "AAAACAAEAAAAAAAEAAMAAQ=="}};
  EXPECT_EQ(ActualResults, ExpectedResults);
}

} // namespace
} // namespace clangd
} // namespace clang
