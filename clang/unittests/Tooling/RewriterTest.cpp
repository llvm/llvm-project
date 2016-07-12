//===- unittest/Tooling/RewriterTest.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RewriterTestContext.h"
#include "clang/Tooling/Core/Replacement.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {
namespace {

TEST(Rewriter, OverwritesChangedFiles) {
  RewriterTestContext Context;
  FileID ID = Context.createOnDiskFile("t.cpp", "line1\nline2\nline3\nline4");
  Context.Rewrite.ReplaceText(Context.getLocation(ID, 2, 1), 5, "replaced");
  EXPECT_FALSE(Context.Rewrite.overwriteChangedFiles());
  EXPECT_EQ("line1\nreplaced\nline3\nline4",
            Context.getFileContentFromDisk("t.cpp")); 
}

TEST(Rewriter, ContinuesOverwritingFilesOnError) {
  RewriterTestContext Context;
  FileID FailingID = Context.createInMemoryFile("invalid/failing.cpp", "test");
  Context.Rewrite.ReplaceText(Context.getLocation(FailingID, 1, 2), 1, "other");
  FileID WorkingID = Context.createOnDiskFile(
    "working.cpp", "line1\nline2\nline3\nline4");
  Context.Rewrite.ReplaceText(Context.getLocation(WorkingID, 2, 1), 5,
                              "replaced");
  EXPECT_TRUE(Context.Rewrite.overwriteChangedFiles());
  EXPECT_EQ("line1\nreplaced\nline3\nline4",
            Context.getFileContentFromDisk("working.cpp")); 
}

TEST(Rewriter, AdjacentInsertAndDelete) {
  Replacements Replaces;
  Replaces.insert(Replacement("<file>", 6, 6, ""));
  Replaces.insert(Replacement("<file>", 6, 0, "replaced\n"));
  auto Rewritten = applyAllReplacements("line1\nline2\nline3\nline4", Replaces);
  EXPECT_TRUE(static_cast<bool>(Rewritten));
  EXPECT_EQ("line1\nreplaced\nline3\nline4", *Rewritten);
}

} // end namespace
} // end namespace tooling
} // end namespace clang
