#include "SyncAPI.h"
#include "TestFS.h"
#include "index/Background.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::_;
using testing::AllOf;
using testing::Not;
using testing::UnorderedElementsAre;

namespace clang {
namespace clangd {

MATCHER_P(Named, N, "") { return arg.Name == N; }
MATCHER(Declared, "") { return !arg.CanonicalDeclaration.FileURI.empty(); }
MATCHER(Defined, "") { return !arg.Definition.FileURI.empty(); }

MATCHER_P(FileURI, F, "") { return arg.Location.FileURI == F; }
testing::Matcher<const RefSlab &>
RefsAre(std::vector<testing::Matcher<Ref>> Matchers) {
  return ElementsAre(testing::Pair(_, UnorderedElementsAreArray(Matchers)));
}

TEST(BackgroundIndexTest, IndexTwoFiles) {
  MockFSProvider FS;
  // a.h yields different symbols when included by A.cc vs B.cc.
  FS.Files[testPath("root/A.h")] = R"cpp(
      void common();
      void f_b();
      #if A
        class A_CC {};
      #else
        class B_CC{};
      #endif
      )cpp";
  FS.Files[testPath("root/A.cc")] =
      "#include \"A.h\"\nvoid g() { (void)common; }";
  FS.Files[testPath("root/B.cc")] =
      R"cpp(
      #define A 0
      #include "A.h"
      void f_b() {
        (void)common;
      })cpp";
  BackgroundIndex Idx(Context::empty(), "", FS, /*URISchmes=*/{"unittest"});

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "-DA=1", testPath("root/A.cc")};
  Idx.enqueue(testPath("root"), Cmd);

  Idx.blockUntilIdleForTest();
  EXPECT_THAT(
      runFuzzyFind(Idx, ""),
      UnorderedElementsAre(Named("common"), Named("A_CC"),
                           AllOf(Named("f_b"), Declared(), Not(Defined()))));

  Cmd.Filename = testPath("root/B.cc");
  Cmd.CommandLine = {"clang++", Cmd.Filename};
  Idx.enqueue(testPath("root"), Cmd);

  Idx.blockUntilIdleForTest();
  // B_CC is dropped as we don't collect symbols from A.h in this compilation.
  EXPECT_THAT(runFuzzyFind(Idx, ""),
              UnorderedElementsAre(Named("common"), Named("A_CC"),
                                   AllOf(Named("f_b"), Declared(), Defined())));

  auto Syms = runFuzzyFind(Idx, "common");
  EXPECT_THAT(Syms, UnorderedElementsAre(Named("common")));
  auto Common = *Syms.begin();
  EXPECT_THAT(getRefs(Idx, Common.ID),
              RefsAre({FileURI("unittest:///root/A.h"),
                       FileURI("unittest:///root/A.cc"),
                       FileURI("unittest:///root/B.cc")}));
}

} // namespace clangd
} // namespace clang
