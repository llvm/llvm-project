//===- unittest/Format/SortIncludesTest.cpp - Include sort unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace {

class SortIncludesTest : public ::testing::Test {
protected:
  std::vector<tooling::Range> GetCodeRange(StringRef Code) {
    return std::vector<tooling::Range>(1, tooling::Range(0, Code.size()));
  }

  std::string sort(StringRef Code, std::vector<tooling::Range> Ranges,
                   StringRef FileName = "input.cc") {
    auto Replaces = sortIncludes(FmtStyle, Code, Ranges, FileName);
    Ranges = tooling::calculateRangesAfterReplacements(Replaces, Ranges);
    auto Sorted = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Sorted));
    auto Result = applyAllReplacements(
        *Sorted, reformat(FmtStyle, *Sorted, Ranges, FileName));
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  std::string sort(StringRef Code, StringRef FileName = "input.cpp") {
    return sort(Code, GetCodeRange(Code), FileName);
  }

  unsigned newCursor(llvm::StringRef Code, unsigned Cursor) {
    sortIncludes(FmtStyle, Code, GetCodeRange(Code), "input.cpp", &Cursor);
    return Cursor;
  }

  FormatStyle FmtStyle = getLLVMStyle();
  tooling::IncludeStyle &Style = FmtStyle.IncludeStyle;
};

TEST_F(SortIncludesTest, BasicSorting) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n"));

  EXPECT_EQ("// comment\n"
            "#include <a>\n"
            "#include <b>\n",
            sort("// comment\n"
                 "#include <b>\n"
                 "#include <a>\n",
                 {tooling::Range(25, 1)}));
}

TEST_F(SortIncludesTest, NoReplacementsForValidIncludes) {
  // Identical #includes have led to a failure with an unstable sort.
  std::string Code = "#include <a>\n"
                     "#include <b>\n"
                     "#include <c>\n"
                     "#include <d>\n"
                     "#include <e>\n"
                     "#include <f>\n";
  EXPECT_TRUE(sortIncludes(FmtStyle, Code, GetCodeRange(Code), "a.cc").empty());
}

TEST_F(SortIncludesTest, SortedIncludesInMultipleBlocksAreMerged) {
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "\n"
                 "#include \"b.h\"\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, SupportClangFormatOff) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n"
            "// clang-format off\n"
            "#include <b>\n"
            "#include <a>\n"
            "#include <c>\n"
            "// clang-format on\n",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "// clang-format off\n"
                 "#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "// clang-format on\n"));
}

TEST_F(SortIncludesTest, IncludeSortingCanBeDisabled) {
  FmtStyle.SortIncludes = false;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "#include \"b.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, MixIncludeAndImport) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#import \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#import \"b.h\"\n"));
}

TEST_F(SortIncludesTest, FixTrailingComments) {
  EXPECT_EQ("#include \"a.h\"  // comment\n"
            "#include \"bb.h\" // comment\n"
            "#include \"ccc.h\"\n",
            sort("#include \"a.h\" // comment\n"
                 "#include \"ccc.h\"\n"
                 "#include \"bb.h\" // comment\n"));
}

TEST_F(SortIncludesTest, LeadingWhitespace) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort(" #include \"a.h\"\n"
                 "  #include \"c.h\"\n"
                 "   #include \"b.h\"\n"));
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("# include \"a.h\"\n"
                 "#  include \"c.h\"\n"
                 "#   include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, GreaterInComment) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\" // >\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\" // >\n"));
}

TEST_F(SortIncludesTest, SortsLocallyInEachBlock) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "\n"
            "#include \"b.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, SortsAllBlocksWhenMerging) {
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, CommentsAlwaysSeparateGroups) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "// comment\n"
            "#include \"b.h\"\n",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "// comment\n"
                 "#include \"b.h\"\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "// comment\n"
            "#include \"b.h\"\n",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "// comment\n"
                 "#include \"b.h\"\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "// comment\n"
            "#include \"b.h\"\n",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "// comment\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, HandlesAngledIncludesAsSeparateBlocks) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "#include <b.h>\n"
            "#include <d.h>\n",
            sort("#include <d.h>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\"\n"));

  FmtStyle = getGoogleStyle(FormatStyle::LK_Cpp);
  EXPECT_EQ("#include <b.h>\n"
            "#include <d.h>\n"
            "#include \"a.h\"\n"
            "#include \"c.h\"\n",
            sort("#include <d.h>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\"\n"));
}

TEST_F(SortIncludesTest, RegroupsAngledIncludesInSeparateBlocks) {
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "\n"
            "#include <b.h>\n"
            "#include <d.h>\n",
            sort("#include <d.h>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\"\n"));
}

TEST_F(SortIncludesTest, HandlesMultilineIncludes) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \\\n"
                 "\"c.h\"\n"
                 "#include \"b.h\"\n"));
}

TEST_F(SortIncludesTest, LeavesMainHeaderFirst) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "a.cc"));
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "a_test.cc"));
  EXPECT_EQ("#include \"llvm/input.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"llvm/input.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "input.mm"));

  // Don't allow prefixes.
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/not_a.h\"\n",
            sort("#include \"llvm/not_a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "a.cc"));

  // Don't do this for _main and other suffixes.
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"\n",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "a_main.cc"));

  // Don't do this in headers.
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"\n",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "a.h"));

  // Only do this in the first #include block.
  EXPECT_EQ("#include <a>\n"
            "\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"\n",
            sort("#include <a>\n"
                 "\n"
                 "#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "a.cc"));

  // Only recognize the first #include with a matching basename as main include.
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"\n",
            sort("#include \"b.h\"\n"
                 "#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"llvm/a.h\"\n",
                 "a.cc"));
}

TEST_F(SortIncludesTest, RecognizeMainHeaderInAllGroups) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;

  EXPECT_EQ("#include \"c.h\"\n"
            "#include \"a.h\"\n"
            "#include \"b.h\"\n",
            sort("#include \"b.h\"\n"
                 "\n"
                 "#include \"a.h\"\n"
                 "#include \"c.h\"\n",
                 "c.cc"));
}

TEST_F(SortIncludesTest, MainHeaderIsSeparatedWhenRegroupping) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;

  EXPECT_EQ("#include \"a.h\"\n"
            "\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"b.h\"\n"
                 "\n"
                 "#include \"a.h\"\n"
                 "#include \"c.h\"\n",
                 "a.cc"));
}

TEST_F(SortIncludesTest, SupportCaseInsensitiveMatching) {
  // Setup an regex for main includes so we can cover those as well.
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";

  // Ensure both main header detection and grouping work in a case insensitive
  // manner.
  EXPECT_EQ("#include \"llvm/A.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"LLVM/z.h\"\n"
            "#include \"llvm/X.h\"\n"
            "#include \"GTest/GTest.h\"\n"
            "#include \"gmock/gmock.h\"\n",
            sort("#include \"c.h\"\n"
                 "#include \"b.h\"\n"
                 "#include \"GTest/GTest.h\"\n"
                 "#include \"llvm/A.h\"\n"
                 "#include \"gmock/gmock.h\"\n"
                 "#include \"llvm/X.h\"\n"
                 "#include \"LLVM/z.h\"\n",
                 "a_TEST.cc"));
}

TEST_F(SortIncludesTest, NegativePriorities) {
  Style.IncludeCategories = {{".*important_os_header.*", -1}, {".*", 1}};
  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "#include \"c_main.h\"\n"
            "#include \"a_other.h\"\n",
            sort("#include \"c_main.h\"\n"
                 "#include \"a_other.h\"\n"
                 "#include \"important_os_header.h\"\n",
                 "c_main.cc"));

  // check stable when re-run
  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "#include \"c_main.h\"\n"
            "#include \"a_other.h\"\n",
            sort("#include \"important_os_header.h\"\n"
                 "#include \"c_main.h\"\n"
                 "#include \"a_other.h\"\n",
                 "c_main.cc"));
}

TEST_F(SortIncludesTest, PriorityGroupsAreSeparatedWhenRegroupping) {
  Style.IncludeCategories = {{".*important_os_header.*", -1}, {".*", 1}};
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;

  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "\n"
            "#include \"c_main.h\"\n"
            "\n"
            "#include \"a_other.h\"\n",
            sort("#include \"c_main.h\"\n"
                 "#include \"a_other.h\"\n"
                 "#include \"important_os_header.h\"\n",
                 "c_main.cc"));

  // check stable when re-run
  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "\n"
            "#include \"c_main.h\"\n"
            "\n"
            "#include \"a_other.h\"\n",
            sort("#include \"important_os_header.h\"\n"
                 "\n"
                 "#include \"c_main.h\"\n"
                 "\n"
                 "#include \"a_other.h\"\n",
                 "c_main.cc"));
}

TEST_F(SortIncludesTest, CalculatesCorrectCursorPosition) {
  std::string Code = "#include <ccc>\n"    // Start of line: 0
                     "#include <bbbbbb>\n" // Start of line: 15
                     "#include <a>\n";     // Start of line: 33
  EXPECT_EQ(31u, newCursor(Code, 0));
  EXPECT_EQ(13u, newCursor(Code, 15));
  EXPECT_EQ(0u, newCursor(Code, 33));

  EXPECT_EQ(41u, newCursor(Code, 10));
  EXPECT_EQ(23u, newCursor(Code, 25));
  EXPECT_EQ(10u, newCursor(Code, 43));
}

TEST_F(SortIncludesTest, DeduplicateIncludes) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <c>\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "#include <c>\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "#include <c>\n"));
}

TEST_F(SortIncludesTest, SortAndDeduplicateIncludes) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <c>\n"
                 "#include <b>\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <c>\n"
                 "#include <b>\n"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <c>\n"
                 "#include <b>\n"));
}

TEST_F(SortIncludesTest, CalculatesCorrectCursorPositionAfterDeduplicate) {
  std::string Code = "#include <b>\n"      // Start of line: 0
                     "#include <a>\n"      // Start of line: 13
                     "#include <b>\n"      // Start of line: 26
                     "#include <b>\n"      // Start of line: 39
                     "#include <c>\n"      // Start of line: 52
                     "#include <b>\n";     // Start of line: 65
  std::string Expected = "#include <a>\n"  // Start of line: 0
                         "#include <b>\n"  // Start of line: 13
                         "#include <c>\n"; // Start of line: 26
  EXPECT_EQ(Expected, sort(Code));
  // Cursor on 'i' in "#include <a>".
  EXPECT_EQ(1u, newCursor(Code, 14));
  // Cursor on 'b' in "#include <b>".
  EXPECT_EQ(23u, newCursor(Code, 10));
  EXPECT_EQ(23u, newCursor(Code, 36));
  EXPECT_EQ(23u, newCursor(Code, 49));
  EXPECT_EQ(23u, newCursor(Code, 36));
  EXPECT_EQ(23u, newCursor(Code, 75));
  // Cursor on '#' in "#include <c>".
  EXPECT_EQ(26u, newCursor(Code, 52));
}

TEST_F(SortIncludesTest, DeduplicateLocallyInEachBlock) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "\n"
            "#include <b>\n"
            "#include <c>\n",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "\n"
                 "#include <c>\n"
                 "#include <b>\n"
                 "#include <b>\n"));
}

TEST_F(SortIncludesTest, ValidAffactedRangesAfterDeduplicatingIncludes) {
  std::string Code = "#include <a>\n"
                     "#include <b>\n"
                     "#include <a>\n"
                     "#include <a>\n"
                     "\n"
                     "   int     x ;";
  std::vector<tooling::Range> Ranges = {tooling::Range(0, 52)};
  auto Replaces = sortIncludes(FmtStyle, Code, Ranges, "input.cpp");
  Ranges = tooling::calculateRangesAfterReplacements(Replaces, Ranges);
  EXPECT_EQ(1u, Ranges.size());
  EXPECT_EQ(0u, Ranges[0].getOffset());
  EXPECT_EQ(26u, Ranges[0].getLength());
}

TEST_F(SortIncludesTest, DoNotSortLikelyXml) {
  EXPECT_EQ("<!--;\n"
            "#include <b>\n"
            "#include <a>\n"
            "-->",
            sort("<!--;\n"
                 "#include <b>\n"
                 "#include <a>\n"
                 "-->"));
}

} // end namespace
} // end namespace format
} // end namespace clang
