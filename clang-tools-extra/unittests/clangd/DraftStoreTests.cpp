//===-- DraftStoreTests.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "DraftStore.h"
#include "SourceCode.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

struct IncrementalTestStep {
  llvm::StringRef Src;
  llvm::StringRef Contents;
};

int rangeLength(llvm::StringRef Code, const Range &Rng) {
  llvm::Expected<size_t> Start = positionToOffset(Code, Rng.start);
  llvm::Expected<size_t> End = positionToOffset(Code, Rng.end);
  assert(Start);
  assert(End);
  return *End - *Start;
}

/// Send the changes one by one to updateDraft, verify the intermediate results.
void stepByStep(llvm::ArrayRef<IncrementalTestStep> Steps) {
  DraftStore DS;
  Annotations InitialSrc(Steps.front().Src);
  constexpr llvm::StringLiteral Path("/hello.cpp");

  // Set the initial content.
  DS.addDraft(Path, InitialSrc.code());

  for (size_t i = 1; i < Steps.size(); i++) {
    Annotations SrcBefore(Steps[i - 1].Src);
    Annotations SrcAfter(Steps[i].Src);
    llvm::StringRef Contents = Steps[i - 1].Contents;
    TextDocumentContentChangeEvent Event{
        SrcBefore.range(),
        rangeLength(SrcBefore.code(), SrcBefore.range()),
        Contents.str(),
    };

    llvm::Expected<std::string> Result = DS.updateDraft(Path, {Event});
    ASSERT_TRUE(!!Result);
    EXPECT_EQ(*Result, SrcAfter.code());
    EXPECT_EQ(*DS.getDraft(Path), SrcAfter.code());
  }
}

/// Send all the changes at once to updateDraft, check only the final result.
void allAtOnce(llvm::ArrayRef<IncrementalTestStep> Steps) {
  DraftStore DS;
  Annotations InitialSrc(Steps.front().Src);
  Annotations FinalSrc(Steps.back().Src);
  constexpr llvm::StringLiteral Path("/hello.cpp");
  std::vector<TextDocumentContentChangeEvent> Changes;

  for (size_t i = 0; i < Steps.size() - 1; i++) {
    Annotations Src(Steps[i].Src);
    llvm::StringRef Contents = Steps[i].Contents;

    Changes.push_back({
        Src.range(),
        rangeLength(Src.code(), Src.range()),
        Contents.str(),
    });
  }

  // Set the initial content.
  DS.addDraft(Path, InitialSrc.code());

  llvm::Expected<std::string> Result = DS.updateDraft(Path, Changes);

  ASSERT_TRUE(!!Result) << llvm::toString(Result.takeError());
  EXPECT_EQ(*Result, FinalSrc.code());
  EXPECT_EQ(*DS.getDraft(Path), FinalSrc.code());
}

TEST(DraftStoreIncrementalUpdateTest, Simple) {
  // clang-format off
  IncrementalTestStep Steps[] =
    {
      // Replace a range
      {
R"cpp(static int
hello[[World]]()
{})cpp",
        "Universe"
      },
      // Delete a range
      {
R"cpp(static int
hello[[Universe]]()
{})cpp",
        ""
      },
      // Add a range
      {
R"cpp(static int
hello[[]]()
{})cpp",
        "Monde"
      },
      {
R"cpp(static int
helloMonde()
{})cpp",
        ""
      }
    };
  // clang-format on

  stepByStep(Steps);
  allAtOnce(Steps);
}

TEST(DraftStoreIncrementalUpdateTest, MultiLine) {
  // clang-format off
  IncrementalTestStep Steps[] =
    {
      // Replace a range
      {
R"cpp(static [[int
helloWorld]]()
{})cpp",
R"cpp(char
welcome)cpp"
      },
      // Delete a range
      {
R"cpp(static char[[
welcome]]()
{})cpp",
        ""
      },
      // Add a range
      {
R"cpp(static char[[]]()
{})cpp",
        R"cpp(
cookies)cpp"
      },
      // Replace the whole file
      {
R"cpp([[static char
cookies()
{}]])cpp",
        R"cpp(#include <stdio.h>
)cpp"
      },
      // Delete the whole file
      {
        R"cpp([[#include <stdio.h>
]])cpp",
        "",
      },
      // Add something to an empty file
      {
        "[[]]",
        R"cpp(int main() {
)cpp",
      },
      {
        R"cpp(int main() {
)cpp",
        ""
      }
    };
  // clang-format on

  stepByStep(Steps);
  allAtOnce(Steps);
}

TEST(DraftStoreIncrementalUpdateTest, WrongRangeLength) {
  DraftStore DS;
  Path File = "foo.cpp";

  DS.addDraft(File, "int main() {}\n");

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 0;
  Change.range->end.character = 2;
  Change.rangeLength = 10;

  Expected<std::string> Result = DS.updateDraft(File, {Change});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(
      toString(Result.takeError()),
      "Change's rangeLength (10) doesn't match the computed range length (2).");
}

TEST(DraftStoreIncrementalUpdateTest, EndBeforeStart) {
  DraftStore DS;
  Path File = "foo.cpp";

  DS.addDraft(File, "int main() {}\n");

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 5;
  Change.range->end.line = 0;
  Change.range->end.character = 3;

  Expected<std::string> Result = DS.updateDraft(File, {Change});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(toString(Result.takeError()),
            "Range's end position (0:3) is before start position (0:5)");
}

TEST(DraftStoreIncrementalUpdateTest, StartCharOutOfRange) {
  DraftStore DS;
  Path File = "foo.cpp";

  DS.addDraft(File, "int main() {}\n");

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 100;
  Change.range->end.line = 0;
  Change.range->end.character = 100;
  Change.text = "foo";

  Expected<std::string> Result = DS.updateDraft(File, {Change});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(toString(Result.takeError()),
            "UTF-16 offset 100 is invalid for line 0");
}

TEST(DraftStoreIncrementalUpdateTest, EndCharOutOfRange) {
  DraftStore DS;
  Path File = "foo.cpp";

  DS.addDraft(File, "int main() {}\n");

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 0;
  Change.range->end.character = 100;
  Change.text = "foo";

  Expected<std::string> Result = DS.updateDraft(File, {Change});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(toString(Result.takeError()),
            "UTF-16 offset 100 is invalid for line 0");
}

TEST(DraftStoreIncrementalUpdateTest, StartLineOutOfRange) {
  DraftStore DS;
  Path File = "foo.cpp";

  DS.addDraft(File, "int main() {}\n");

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 100;
  Change.range->start.character = 0;
  Change.range->end.line = 100;
  Change.range->end.character = 0;
  Change.text = "foo";

  Expected<std::string> Result = DS.updateDraft(File, {Change});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(toString(Result.takeError()), "Line value is out of range (100)");
}

TEST(DraftStoreIncrementalUpdateTest, EndLineOutOfRange) {
  DraftStore DS;
  Path File = "foo.cpp";

  DS.addDraft(File, "int main() {}\n");

  TextDocumentContentChangeEvent Change;
  Change.range.emplace();
  Change.range->start.line = 0;
  Change.range->start.character = 0;
  Change.range->end.line = 100;
  Change.range->end.character = 0;
  Change.text = "foo";

  Expected<std::string> Result = DS.updateDraft(File, {Change});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(toString(Result.takeError()), "Line value is out of range (100)");
}

/// Check that if a valid change is followed by an invalid change, the original
/// version of the document (prior to all changes) is kept.
TEST(DraftStoreIncrementalUpdateTest, InvalidRangeInASequence) {
  DraftStore DS;
  Path File = "foo.cpp";

  StringRef OriginalContents = "int main() {}\n";
  DS.addDraft(File, OriginalContents);

  // The valid change
  TextDocumentContentChangeEvent Change1;
  Change1.range.emplace();
  Change1.range->start.line = 0;
  Change1.range->start.character = 0;
  Change1.range->end.line = 0;
  Change1.range->end.character = 0;
  Change1.text = "Hello ";

  // The invalid change
  TextDocumentContentChangeEvent Change2;
  Change2.range.emplace();
  Change2.range->start.line = 0;
  Change2.range->start.character = 5;
  Change2.range->end.line = 0;
  Change2.range->end.character = 100;
  Change2.text = "something";

  Expected<std::string> Result = DS.updateDraft(File, {Change1, Change2});

  EXPECT_TRUE(!Result);
  EXPECT_EQ(toString(Result.takeError()),
            "UTF-16 offset 100 is invalid for line 0");

  Optional<std::string> Contents = DS.getDraft(File);
  EXPECT_TRUE(Contents);
  EXPECT_EQ(*Contents, OriginalContents);
}

} // namespace
} // namespace clangd
} // namespace clang
