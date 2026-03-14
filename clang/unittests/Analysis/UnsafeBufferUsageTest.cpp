#include "gtest/gtest.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Analysis/Analyses/UnsafeBufferUsage.h"

using namespace clang;

namespace {
// The test fixture.
class UnsafeBufferUsageTest : public ::testing::Test {
protected:
  UnsafeBufferUsageTest()
      : FileMgr(FileMgrOpts),
        Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr) {}

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
};
} // namespace

TEST_F(UnsafeBufferUsageTest, FixItHintsConflict) {
  FileEntryRef DummyFile = FileMgr.getVirtualFileRef("<virtual>", 100, 0);
  FileID DummyFileID = SourceMgr.getOrCreateFileID(DummyFile, SrcMgr::C_User);
  SourceLocation StartLoc = SourceMgr.getLocForStartOfFile(DummyFileID);

#define MkDummyHint(Begin, End)                                                \
  FixItHint::CreateReplacement(SourceRange(StartLoc.getLocWithOffset((Begin)), \
                                           StartLoc.getLocWithOffset((End))),  \
                               "dummy")

  FixItHint H1 = MkDummyHint(0, 5);
  FixItHint H2 = MkDummyHint(6, 10);
  FixItHint H3 = MkDummyHint(20, 25);
  llvm::SmallVector<FixItHint> Fixes;

  // Test non-overlapping fix-its:
  Fixes = {H1, H2, H3};
  EXPECT_FALSE(internal::anyConflict(Fixes, SourceMgr));
  Fixes = {H3, H2, H1}; // re-order
  EXPECT_FALSE(internal::anyConflict(Fixes, SourceMgr));

  // Test overlapping fix-its:
  Fixes = {H1, H2, H3, MkDummyHint(0, 4) /* included in H1 */};
  EXPECT_TRUE(internal::anyConflict(Fixes, SourceMgr));

  Fixes = {H1, H2, H3, MkDummyHint(10, 15) /* overlaps H2 */};
  EXPECT_TRUE(internal::anyConflict(Fixes, SourceMgr));

  Fixes = {H1, H2, H3, MkDummyHint(7, 23) /* overlaps H2, H3 */};
  EXPECT_TRUE(internal::anyConflict(Fixes, SourceMgr));

  Fixes = {H1, H2, H3, MkDummyHint(2, 23) /* overlaps H1, H2, and H3 */};
  EXPECT_TRUE(internal::anyConflict(Fixes, SourceMgr));
}
