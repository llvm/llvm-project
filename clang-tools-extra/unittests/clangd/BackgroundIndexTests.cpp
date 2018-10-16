#include "SyncAPI.h"
#include "TestFS.h"
#include "index/Background.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using testing::UnorderedElementsAre;

namespace clang {
namespace clangd {

MATCHER_P(Named, N, "") { return arg.Name == N; }

TEST(BackgroundIndexTest, IndexTwoFiles) {
  MockFSProvider FS;
  // a.h yields different symbols when included by A.cc vs B.cc.
  // Currently we store symbols for each TU, so we get both.
  FS.Files[testPath("root/A.h")] = "void a_h(); void NAME(){}";
  FS.Files[testPath("root/A.cc")] = "#include \"A.h\"";
  FS.Files[testPath("root/B.cc")] = "#define NAME bar\n#include \"A.h\"";
  BackgroundIndex Idx(Context::empty(), "", FS);

  tooling::CompileCommand Cmd;
  Cmd.Filename = testPath("root/A.cc");
  Cmd.Directory = testPath("root");
  Cmd.CommandLine = {"clang++", "-DNAME=foo", testPath("root/A.cc")};
  Idx.enqueue(testPath("root"), Cmd);
  Cmd.CommandLine.back() = Cmd.Filename = testPath("root/B.cc");
  Idx.enqueue(testPath("root"), Cmd);

  Idx.blockUntilIdleForTest();
  EXPECT_THAT(runFuzzyFind(Idx, ""),
              UnorderedElementsAre(Named("a_h"), Named("foo"), Named("bar")));
}

} // namespace clangd
} // namespace clang
