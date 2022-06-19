//===-- SemanticSelectionTests.cpp  ----------------*- C++ -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdServer.h"
#include "Protocol.h"
#include "SemanticSelection.h"
#include "SyncAPI.h"
#include "TestFS.h"
#include "TestTU.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace clangd {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

// front() is SR.range, back() is outermost range.
std::vector<Range> gatherRanges(const SelectionRange &SR) {
  std::vector<Range> Ranges;
  for (const SelectionRange *S = &SR; S; S = S->parent.get())
    Ranges.push_back(S->range);
  return Ranges;
}

std::vector<Range>
gatherFoldingRanges(llvm::ArrayRef<FoldingRange> FoldingRanges) {
  std::vector<Range> Ranges;
  Range NextRange;
  for (const auto &R : FoldingRanges) {
    NextRange.start.line = R.startLine;
    NextRange.start.character = R.startCharacter;
    NextRange.end.line = R.endLine;
    NextRange.end.character = R.endCharacter;
    Ranges.push_back(NextRange);
  }
  return Ranges;
}

TEST(SemanticSelection, All) {
  const char *Tests[] = {
      R"cpp( // Single statement in a function body.
        [[void func() [[{
          [[[[int v = [[1^00]]]];]]
        }]]]]
      )cpp",
      R"cpp( // Expression
        [[void func() [[{
          int a = 1;
          // int v = (10 + 2) * (a + a);
          [[[[int v = [[[[([[[[10^]] + 2]])]] * (a + a)]]]];]]
        }]]]]
      )cpp",
      R"cpp( // Function call.
        int add(int x, int y) { return x + y; }
        [[void callee() [[{
          // int res = add(11, 22);
          [[[[int res = [[add([[1^1]], 22)]]]];]]
        }]]]]
      )cpp",
      R"cpp( // Tricky macros.
        #define MUL ) * (
        [[void func() [[{
          // int var = (4 + 15 MUL 6 + 10);
          [[[[int var = [[[[([[4 + [[1^5]]]] MUL]] 6 + 10)]]]];]]
        }]]]]
       )cpp",
      R"cpp( // Cursor inside a macro.
        #define HASH(x) ((x) % 10)
        [[void func() [[{
          [[[[int a = [[HASH([[[[2^3]] + 34]])]]]];]]
        }]]]]
       )cpp",
      R"cpp( // Cursor on a macro.
        #define HASH(x) ((x) % 10)
        [[void func() [[{
          [[[[int a = [[HA^SH(23)]]]];]]
        }]]]]
       )cpp",
      R"cpp( // Multiple declaration.
        [[void func() [[{
          [[[[int var1, var^2]], var3;]]
        }]]]]
       )cpp",
      R"cpp( // Before comment.
        [[void func() [[{
          int var1 = 1;
          [[[[int var2 = [[[[var1]]^ /*some comment*/ + 41]]]];]]
        }]]]]
       )cpp",
      // Empty file.
      "[[^]]",
      // FIXME: We should get the whole DeclStmt as a range.
      R"cpp( // Single statement in TU.
        [[int v = [[1^00]]]];
      )cpp",
      R"cpp( // Cursor at end of VarDecl.
        [[int v = [[100]]^]];
      )cpp",
      // FIXME: No node found associated to the position.
      R"cpp( // Cursor in between spaces.
        void func() {
          int v = 100 + [[^]]  100;
        }
      )cpp",
      // Structs.
      R"cpp(
        struct AAA { struct BBB { static int ccc(); };};
        [[void func() [[{
          // int x = AAA::BBB::ccc();
          [[[[int x = [[[[AAA::BBB::c^cc]]()]]]];]]
        }]]]]
      )cpp",
      R"cpp(
        struct AAA { struct BBB { static int ccc(); };};
        [[void func() [[{
          // int x = AAA::BBB::ccc();
          [[[[int x = [[[[[[[[[[AA^A]]::]]BBB::]]ccc]]()]]]];]]
        }]]]]
      )cpp",
      R"cpp( // Inside struct.
        struct A { static int a(); };
        [[struct B {
          [[static int b() [[{
            [[return [[[[1^1]] + 2]]]];
          }]]]]
        }]];
      )cpp",
      // Namespaces.
      R"cpp(
        [[namespace nsa {
          [[namespace nsb {
            static int ccc();
            [[void func() [[{
              // int x = nsa::nsb::ccc();
              [[[[int x = [[[[nsa::nsb::cc^c]]()]]]];]]
            }]]]]
          }]]
        }]]
      )cpp",

  };

  for (const char *Test : Tests) {
    auto T = Annotations(Test);
    auto AST = TestTU::withCode(T.code()).build();
    EXPECT_THAT(gatherRanges(llvm::cantFail(getSemanticRanges(AST, T.point()))),
                ElementsAreArray(T.ranges()))
        << Test;
  }
}

TEST(SemanticSelection, RunViaClangdServer) {
  MockFS FS;
  MockCompilationDatabase CDB;
  ClangdServer Server(CDB, FS, ClangdServer::optsForTest());

  auto FooH = testPath("foo.h");
  FS.Files[FooH] = R"cpp(
    int foo(int x);
    #define HASH(x) ((x) % 10)
  )cpp";

  auto FooCpp = testPath("Foo.cpp");
  const char *SourceContents = R"cpp(
  #include "foo.h"
  [[void bar(int& inp) [[{
    // inp = HASH(foo(inp));
    [[inp = [[HASH([[foo([[in^p]])]])]]]];
  }]]]]
  $empty[[^]]
  )cpp";
  Annotations SourceAnnotations(SourceContents);
  FS.Files[FooCpp] = std::string(SourceAnnotations.code());
  Server.addDocument(FooCpp, SourceAnnotations.code());

  auto Ranges = runSemanticRanges(Server, FooCpp, SourceAnnotations.points());
  ASSERT_TRUE(bool(Ranges))
      << "getSemanticRange returned an error: " << Ranges.takeError();
  ASSERT_EQ(Ranges->size(), SourceAnnotations.points().size());
  EXPECT_THAT(gatherRanges(Ranges->front()),
              ElementsAreArray(SourceAnnotations.ranges()));
  EXPECT_THAT(gatherRanges(Ranges->back()),
              ElementsAre(SourceAnnotations.range("empty")));
}

TEST(FoldingRanges, All) {
  const char *Tests[] = {
      R"cpp(
        #define FOO int foo() {\
          int Variable = 42; \
        }

        // Do not generate folding range for braces within macro expansion.
        FOO

        // Do not generate folding range within macro arguments.
        #define FUNCTOR(functor) functor
        void func() {[[
          FUNCTOR([](){});
        ]]}

        // Do not generate folding range with a brace coming from macro.
        #define LBRACE {
        void bar() LBRACE
          int X = 42;
        }
      )cpp",
      R"cpp(
        void func() {[[
          int Variable = 100;

          if (Variable > 5) {[[
            Variable += 42;
          ]]} else if (Variable++)
            ++Variable;
          else {[[
            Variable--;
          ]]}

          // Do not generate FoldingRange for empty CompoundStmts.
          for (;;) {}

          // If there are newlines between {}, we should generate one.
          for (;;) {[[

          ]]}
        ]]}
      )cpp",
      R"cpp(
        class Foo {
        public:
          Foo() {[[
            int X = 1;
          ]]}

        private:
          int getBar() {[[
            return 42;
          ]]}

          // Braces are located at the same line: no folding range here.
          void getFooBar() { }
        };
      )cpp",
  };
  for (const char *Test : Tests) {
    auto T = Annotations(Test);
    auto AST = TestTU::withCode(T.code()).build();
    EXPECT_THAT(gatherFoldingRanges(llvm::cantFail(getFoldingRanges(AST))),
                UnorderedElementsAreArray(T.ranges()))
        << Test;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
