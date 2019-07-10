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
      {HighlightingKind::Enum, "Enum"}};
  std::vector<HighlightingToken> ExpectedTokens;
  for (const auto &KindString : KindToString) {
    std::vector<HighlightingToken> Toks = makeHighlightingTokens(
        Test.ranges(KindString.second), KindString.first);
    ExpectedTokens.insert(ExpectedTokens.end(), Toks.begin(), Toks.end());
  }

  auto ActualTokens = getSemanticHighlightings(AST);
  EXPECT_THAT(ActualTokens, testing::UnorderedElementsAreArray(ExpectedTokens));
}

TEST(SemanticHighlighting, GetsCorrectTokens) {
  const char *TestCases[] = {
    R"cpp(
      struct $Class[[AS]] {
        double SomeMember;
      };
      struct {
      } $Variable[[S]];
      void $Function[[foo]](int $Variable[[A]], $Class[[AS]] $Variable[[As]]) {
        auto $Variable[[VeryLongVariableName]] = 12312;
        $Class[[AS]]     $Variable[[AA]];
        auto $Variable[[L]] = $Variable[[AA]].SomeMember + $Variable[[A]];
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
        void $Function[[abc]]();
      };
    )cpp",
    R"cpp(
      namespace abc {
        template<typename T>
        struct $Class[[A]] {
          T t;
        };
      }
      template<typename T>
      struct $Class[[C]] : abc::A<T> {
        typename T::A* D;
      };
      abc::$Class[[A]]<int> $Variable[[AA]];
      typedef abc::$Class[[A]]<int> AAA;
      struct $Class[[B]] {
        $Class[[B]]();
        ~$Class[[B]]();
        void operator<<($Class[[B]]);
        $Class[[AAA]] AA;
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
      enum class $Enum[[E]] {};
      enum $Enum[[EE]] {};
      struct $Class[[A]] {
        $Enum[[E]] EEE;
        $Enum[[EE]] EEEE;
      };
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
