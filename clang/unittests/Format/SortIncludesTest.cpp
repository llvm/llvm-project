//===- unittest/Format/SortIncludesTest.cpp - Include sort unit tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/ADT/StringRef.h"
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
                   StringRef FileName = "input.cc",
                   unsigned ExpectedNumRanges = 1) {
    auto Replaces = sortIncludes(FmtStyle, Code, Ranges, FileName);
    Ranges = tooling::calculateRangesAfterReplacements(Replaces, Ranges);
    EXPECT_EQ(ExpectedNumRanges, Replaces.size());
    auto Sorted = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Sorted));
    auto Result = applyAllReplacements(
        *Sorted, reformat(FmtStyle, *Sorted, Ranges, FileName));
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  std::string sort(StringRef Code, StringRef FileName = "input.cpp",
                   unsigned ExpectedNumRanges = 1) {
    return sort(Code, GetCodeRange(Code), FileName, ExpectedNumRanges);
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
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\""));

  EXPECT_EQ("// comment\n"
            "#include <a>\n"
            "#include <b>",
            sort("// comment\n"
                 "#include <b>\n"
                 "#include <a>",
                 {tooling::Range(25, 1)}));
}

TEST_F(SortIncludesTest, TrailingComments) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\" /* long\n"
            "                  * long\n"
            "                  * comment*/\n"
            "#include \"c.h\"\n"
            "#include \"d.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\" /* long\n"
                 "                  * long\n"
                 "                  * comment*/\n"
                 "#include \"d.h\""));
}

TEST_F(SortIncludesTest, SortedIncludesUsingSortPriorityAttribute) {
  FmtStyle.IncludeStyle.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  FmtStyle.IncludeStyle.IncludeCategories = {
      {"^<sys/param\\.h>", 1, 0, false},
      {"^<sys/types\\.h>", 1, 1, false},
      {"^<sys.*/", 1, 2, false},
      {"^<uvm/", 2, 3, false},
      {"^<machine/", 3, 4, false},
      {"^<dev/", 4, 5, false},
      {"^<net.*/", 5, 6, false},
      {"^<protocols/", 5, 7, false},
      {"^<(fs|miscfs|msdosfs|nfs|ntfs|ufs)/", 6, 8, false},
      {"^<(x86|amd64|i386|xen)/", 7, 8, false},
      {"<path", 9, 11, false},
      {"^<[^/].*\\.h>", 8, 10, false},
      {"^\".*\\.h\"", 10, 12, false}};
  EXPECT_EQ("#include <sys/param.h>\n"
            "#include <sys/types.h>\n"
            "#include <sys/ioctl.h>\n"
            "#include <sys/socket.h>\n"
            "#include <sys/stat.h>\n"
            "#include <sys/wait.h>\n"
            "\n"
            "#include <net/if.h>\n"
            "#include <net/if_dl.h>\n"
            "#include <net/route.h>\n"
            "#include <netinet/in.h>\n"
            "#include <protocols/rwhod.h>\n"
            "\n"
            "#include <assert.h>\n"
            "#include <errno.h>\n"
            "#include <inttypes.h>\n"
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            "\n"
            "#include <paths.h>\n"
            "\n"
            "#include \"pathnames.h\"",
            sort("#include <sys/param.h>\n"
                 "#include <sys/types.h>\n"
                 "#include <sys/ioctl.h>\n"
                 "#include <net/if_dl.h>\n"
                 "#include <net/route.h>\n"
                 "#include <netinet/in.h>\n"
                 "#include <sys/socket.h>\n"
                 "#include <sys/stat.h>\n"
                 "#include <sys/wait.h>\n"
                 "#include <net/if.h>\n"
                 "#include <protocols/rwhod.h>\n"
                 "#include <assert.h>\n"
                 "#include <paths.h>\n"
                 "#include \"pathnames.h\"\n"
                 "#include <errno.h>\n"
                 "#include <inttypes.h>\n"
                 "#include <stdio.h>\n"
                 "#include <stdlib.h>"));
}
TEST_F(SortIncludesTest, SortPriorityNotDefined) {
  FmtStyle = getLLVMStyle();
  EXPECT_EQ("#include \"FormatTestUtils.h\"\n"
            "#include \"clang/Format/Format.h\"\n"
            "#include \"llvm/ADT/None.h\"\n"
            "#include \"llvm/Support/Debug.h\"\n"
            "#include \"gtest/gtest.h\"",
            sort("#include \"clang/Format/Format.h\"\n"
                 "#include \"llvm/ADT/None.h\"\n"
                 "#include \"FormatTestUtils.h\"\n"
                 "#include \"gtest/gtest.h\"\n"
                 "#include \"llvm/Support/Debug.h\""));
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

TEST_F(SortIncludesTest, MainFileHeader) {
  std::string Code = "#include <string>\n"
                     "\n"
                     "#include \"a/extra_action.proto.h\"\n";
  FmtStyle = getGoogleStyle(FormatStyle::LK_Cpp);
  EXPECT_TRUE(
      sortIncludes(FmtStyle, Code, GetCodeRange(Code), "a/extra_action.cc")
          .empty());

  EXPECT_EQ("#include \"foo.bar.h\"\n"
            "\n"
            "#include \"a.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"foo.bar.h\"",
                 "foo.bar.cc"));
}

TEST_F(SortIncludesTest, SortedIncludesInMultipleBlocksAreMerged) {
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "\n"
                 "#include \"b.h\""));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "\n"
                 "#include \"b.h\""));
}

TEST_F(SortIncludesTest, SupportClangFormatOff) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n"
            "// clang-format off\n"
            "#include <b>\n"
            "#include <a>\n"
            "#include <c>\n"
            "// clang-format on",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "// clang-format off\n"
                 "#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "// clang-format on"));

  Style.IncludeBlocks = Style.IBS_Merge;
  std::string Code = "// clang-format off\r\n"
                     "#include \"d.h\"\r\n"
                     "#include \"b.h\"\r\n"
                     "// clang-format on\r\n"
                     "\r\n"
                     "#include \"c.h\"\r\n"
                     "#include \"a.h\"\r\n"
                     "#include \"e.h\"\r\n";

  std::string Expected = "// clang-format off\r\n"
                         "#include \"d.h\"\r\n"
                         "#include \"b.h\"\r\n"
                         "// clang-format on\r\n"
                         "\r\n"
                         "#include \"e.h\"\r\n"
                         "#include \"a.h\"\r\n"
                         "#include \"c.h\"\r\n";

  EXPECT_EQ(Expected, sort(Code, "e.cpp", 1));
}

TEST_F(SortIncludesTest, SupportClangFormatOffCStyle) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n"
            "/* clang-format off */\n"
            "#include <b>\n"
            "#include <a>\n"
            "#include <c>\n"
            "/* clang-format on */",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "/* clang-format off */\n"
                 "#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "/* clang-format on */"));

  // Not really turning it off
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>\n"
            "/* clang-format offically */\n"
            "#include <a>\n"
            "#include <b>\n"
            "#include <c>\n"
            "/* clang-format onwards */",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "/* clang-format offically */\n"
                 "#include <b>\n"
                 "#include <a>\n"
                 "#include <c>\n"
                 "/* clang-format onwards */",
                 "input.h", 2));
}

TEST_F(SortIncludesTest, IncludeSortingCanBeDisabled) {
  FmtStyle.SortIncludes = FormatStyle::SI_Never;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "#include \"b.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "input.h", 0));
}

TEST_F(SortIncludesTest, MixIncludeAndImport) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#import \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#import \"b.h\""));
}

TEST_F(SortIncludesTest, FixTrailingComments) {
  EXPECT_EQ("#include \"a.h\"  // comment\n"
            "#include \"bb.h\" // comment\n"
            "#include \"ccc.h\"",
            sort("#include \"a.h\" // comment\n"
                 "#include \"ccc.h\"\n"
                 "#include \"bb.h\" // comment"));
}

TEST_F(SortIncludesTest, LeadingWhitespace) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort(" #include \"a.h\"\n"
                 "  #include \"c.h\"\n"
                 "   #include \"b.h\""));
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("# include \"a.h\"\n"
                 "#  include \"c.h\"\n"
                 "#   include \"b.h\""));
  EXPECT_EQ("#include \"a.h\"", sort("#include \"a.h\"\n"
                                     " #include \"a.h\""));
}

TEST_F(SortIncludesTest, TrailingWhitespace) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"a.h\" \n"
                 "#include \"c.h\"  \n"
                 "#include \"b.h\"   "));
  EXPECT_EQ("#include \"a.h\"", sort("#include \"a.h\"\n"
                                     "#include \"a.h\" "));
}

TEST_F(SortIncludesTest, GreaterInComment) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\" // >\n"
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\" // >"));
}

TEST_F(SortIncludesTest, SortsLocallyInEachBlock) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "\n"
            "#include \"b.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "#include \"b.h\"",
                 "input.h", 0));
}

TEST_F(SortIncludesTest, SortsAllBlocksWhenMerging) {
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "\n"
                 "#include \"b.h\""));
}

TEST_F(SortIncludesTest, CommentsAlwaysSeparateGroups) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "// comment\n"
            "#include \"b.h\"",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "// comment\n"
                 "#include \"b.h\""));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "// comment\n"
            "#include \"b.h\"",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "// comment\n"
                 "#include \"b.h\""));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "// comment\n"
            "#include \"b.h\"",
            sort("#include \"c.h\"\n"
                 "#include \"a.h\"\n"
                 "// comment\n"
                 "#include \"b.h\""));
}

TEST_F(SortIncludesTest, HandlesAngledIncludesAsSeparateBlocks) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "#include <array>\n"
            "#include <b.h>\n"
            "#include <d.h>\n"
            "#include <vector>",
            sort("#include <vector>\n"
                 "#include <d.h>\n"
                 "#include <array>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\""));

  FmtStyle = getGoogleStyle(FormatStyle::LK_Cpp);
  EXPECT_EQ("#include <b.h>\n"
            "#include <d.h>\n"
            "\n"
            "#include <array>\n"
            "#include <vector>\n"
            "\n"
            "#include \"a.h\"\n"
            "#include \"c.h\"",
            sort("#include <vector>\n"
                 "#include <d.h>\n"
                 "#include <array>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\""));
}

TEST_F(SortIncludesTest, RegroupsAngledIncludesInSeparateBlocks) {
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "\n"
            "#include <b.h>\n"
            "#include <d.h>",
            sort("#include <d.h>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\""));
}

TEST_F(SortIncludesTest, HandlesMultilineIncludes) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"a.h\"\n"
                 "#include \\\n"
                 "\"c.h\"\n"
                 "#include \"b.h\""));
}

TEST_F(SortIncludesTest, HandlesTrailingCommentsWithAngleBrackets) {
  // Regression test from the discussion at https://reviews.llvm.org/D121370.
  EXPECT_EQ("#include <cstdint>\n"
            "\n"
            "#include \"util/bar.h\"\n"
            "#include \"util/foo/foo.h\" // foo<T>",
            sort("#include <cstdint>\n"
                 "\n"
                 "#include \"util/bar.h\"\n"
                 "#include \"util/foo/foo.h\" // foo<T>",
                 /*FileName=*/"input.cc",
                 /*ExpectedNumRanges=*/0));
}

TEST_F(SortIncludesTest, LeavesMainHeaderFirst) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a.cc"));
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a_test.cc"));
  EXPECT_EQ("#include \"llvm/input.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/input.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "input.mm"));

  // Don't allow prefixes.
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/not_a.h\"",
            sort("#include \"llvm/not_a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a.cc"));

  // Don't do this for _main and other suffixes.
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a_main.cc"));

  // Don't do this in headers.
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a.h"));

  // Only do this in the first #include block.
  EXPECT_EQ("#include <a>\n"
            "\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"",
            sort("#include <a>\n"
                 "\n"
                 "#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a.cc"));

  // Only recognize the first #include with a matching basename as main include.
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"",
            sort("#include \"b.h\"\n"
                 "#include \"a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"llvm/a.h\"",
                 "a.cc"));
}

TEST_F(SortIncludesTest, LeavesMainHeaderFirstInAdditionalExtensions) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?|(Impl)?$";
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a_test.xxx"));
  EXPECT_EQ("#include \"b.h\"\n"
            "#include \"c.h\"\n"
            "#include \"llvm/a.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "aImpl.hpp"));

  // .cpp extension is considered "main" by default
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "aImpl.cpp"));
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a_test.cpp"));

  // Allow additional filenames / extensions
  Style.IncludeIsMainSourceRegex = "(Impl\\.hpp)|(\\.xxx)$";
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "a_test.xxx"));
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"llvm/a.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"",
                 "aImpl.hpp"));
}

TEST_F(SortIncludesTest, RecognizeMainHeaderInAllGroups) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;

  EXPECT_EQ("#include \"c.h\"\n"
            "#include \"a.h\"\n"
            "#include \"b.h\"",
            sort("#include \"b.h\"\n"
                 "\n"
                 "#include \"a.h\"\n"
                 "#include \"c.h\"",
                 "c.cc"));
}

TEST_F(SortIncludesTest, MainHeaderIsSeparatedWhenRegroupping) {
  Style.IncludeIsMainRegex = "([-_](test|unittest))?$";
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;

  EXPECT_EQ("#include \"a.h\"\n"
            "\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"",
            sort("#include \"b.h\"\n"
                 "\n"
                 "#include \"a.h\"\n"
                 "#include \"c.h\"",
                 "a.cc"));
}

TEST_F(SortIncludesTest, SupportOptionalCaseSensitiveSorting) {
  EXPECT_FALSE(FmtStyle.SortIncludes == FormatStyle::SI_CaseInsensitive);

  FmtStyle.SortIncludes = FormatStyle::SI_CaseInsensitive;

  EXPECT_EQ("#include \"A/B.h\"\n"
            "#include \"A/b.h\"\n"
            "#include \"a/b.h\"\n"
            "#include \"B/A.h\"\n"
            "#include \"B/a.h\"",
            sort("#include \"B/a.h\"\n"
                 "#include \"B/A.h\"\n"
                 "#include \"A/B.h\"\n"
                 "#include \"a/b.h\"\n"
                 "#include \"A/b.h\"",
                 "a.h"));

  Style.IncludeBlocks = clang::tooling::IncludeStyle::IBS_Regroup;
  Style.IncludeCategories = {
      {"^\"", 1, 0, false}, {"^<.*\\.h>$", 2, 0, false}, {"^<", 3, 0, false}};

  StringRef UnsortedCode = "#include \"qt.h\"\n"
                           "#include <algorithm>\n"
                           "#include <qtwhatever.h>\n"
                           "#include <Qtwhatever.h>\n"
                           "#include <Algorithm>\n"
                           "#include \"vlib.h\"\n"
                           "#include \"Vlib.h\"\n"
                           "#include \"AST.h\"";

  EXPECT_EQ("#include \"AST.h\"\n"
            "#include \"qt.h\"\n"
            "#include \"Vlib.h\"\n"
            "#include \"vlib.h\"\n"
            "\n"
            "#include <Qtwhatever.h>\n"
            "#include <qtwhatever.h>\n"
            "\n"
            "#include <Algorithm>\n"
            "#include <algorithm>",
            sort(UnsortedCode));
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
            "#include \"gmock/gmock.h\"",
            sort("#include \"c.h\"\n"
                 "#include \"b.h\"\n"
                 "#include \"GTest/GTest.h\"\n"
                 "#include \"llvm/A.h\"\n"
                 "#include \"gmock/gmock.h\"\n"
                 "#include \"llvm/X.h\"\n"
                 "#include \"LLVM/z.h\"",
                 "a_TEST.cc"));
}

TEST_F(SortIncludesTest, SupportOptionalCaseSensitiveMachting) {
  Style.IncludeBlocks = clang::tooling::IncludeStyle::IBS_Regroup;
  Style.IncludeCategories = {{"^\"", 1, 0, false},
                             {"^<.*\\.h>$", 2, 0, false},
                             {"^<Q[A-Z][^\\.]*>", 3, 0, false},
                             {"^<Qt[^\\.]*>", 4, 0, false},
                             {"^<", 5, 0, false}};

  StringRef UnsortedCode = "#include <QWidget>\n"
                           "#include \"qt.h\"\n"
                           "#include <algorithm>\n"
                           "#include <windows.h>\n"
                           "#include <QLabel>\n"
                           "#include \"qa.h\"\n"
                           "#include <queue>\n"
                           "#include <qtwhatever.h>\n"
                           "#include <QtGlobal>";

  EXPECT_EQ("#include \"qa.h\"\n"
            "#include \"qt.h\"\n"
            "\n"
            "#include <qtwhatever.h>\n"
            "#include <windows.h>\n"
            "\n"
            "#include <QLabel>\n"
            "#include <QWidget>\n"
            "#include <QtGlobal>\n"
            "#include <queue>\n"
            "\n"
            "#include <algorithm>",
            sort(UnsortedCode));

  Style.IncludeCategories[2].RegexIsCaseSensitive = true;
  Style.IncludeCategories[3].RegexIsCaseSensitive = true;
  EXPECT_EQ("#include \"qa.h\"\n"
            "#include \"qt.h\"\n"
            "\n"
            "#include <qtwhatever.h>\n"
            "#include <windows.h>\n"
            "\n"
            "#include <QLabel>\n"
            "#include <QWidget>\n"
            "\n"
            "#include <QtGlobal>\n"
            "\n"
            "#include <algorithm>\n"
            "#include <queue>",
            sort(UnsortedCode));
}

TEST_F(SortIncludesTest, NegativePriorities) {
  Style.IncludeCategories = {{".*important_os_header.*", -1, 0, false},
                             {".*", 1, 0, false}};
  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "#include \"c_main.h\"\n"
            "#include \"a_other.h\"",
            sort("#include \"c_main.h\"\n"
                 "#include \"a_other.h\"\n"
                 "#include \"important_os_header.h\"",
                 "c_main.cc"));

  // check stable when re-run
  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "#include \"c_main.h\"\n"
            "#include \"a_other.h\"",
            sort("#include \"important_os_header.h\"\n"
                 "#include \"c_main.h\"\n"
                 "#include \"a_other.h\"",
                 "c_main.cc", 0));
}

TEST_F(SortIncludesTest, PriorityGroupsAreSeparatedWhenRegroupping) {
  Style.IncludeCategories = {{".*important_os_header.*", -1, 0, false},
                             {".*", 1, 0, false}};
  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;

  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "\n"
            "#include \"c_main.h\"\n"
            "\n"
            "#include \"a_other.h\"",
            sort("#include \"c_main.h\"\n"
                 "#include \"a_other.h\"\n"
                 "#include \"important_os_header.h\"",
                 "c_main.cc"));

  // check stable when re-run
  EXPECT_EQ("#include \"important_os_header.h\"\n"
            "\n"
            "#include \"c_main.h\"\n"
            "\n"
            "#include \"a_other.h\"",
            sort("#include \"important_os_header.h\"\n"
                 "\n"
                 "#include \"c_main.h\"\n"
                 "\n"
                 "#include \"a_other.h\"",
                 "c_main.cc", 0));
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

TEST_F(SortIncludesTest, CalculatesCorrectCursorPositionWithRegrouping) {
  Style.IncludeBlocks = Style.IBS_Regroup;
  std::string Code = "#include \"b\"\n"      // Start of line: 0
                     "\n"                    // Start of line: 13
                     "#include \"aa\"\n"     // Start of line: 14
                     "int i;";               // Start of line: 28
  std::string Expected = "#include \"aa\"\n" // Start of line: 0
                         "#include \"b\"\n"  // Start of line: 14
                         "int i;";           // Start of line: 27
  EXPECT_EQ(Expected, sort(Code));
  EXPECT_EQ(12u, newCursor(Code, 26)); // Closing quote of "aa"
  EXPECT_EQ(26u, newCursor(Code, 27)); // Newline after "aa"
  EXPECT_EQ(27u, newCursor(Code, 28)); // Start of last line
}

TEST_F(SortIncludesTest, DeduplicateIncludes) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <c>"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "#include <c>"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <b>\n"
                 "#include <c>"));
}

TEST_F(SortIncludesTest, SortAndDeduplicateIncludes) {
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "#include <b>\n"
                 "#include <b>\n"
                 "#include <c>\n"
                 "#include <b>"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Merge;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <c>\n"
                 "#include <b>"));

  Style.IncludeBlocks = tooling::IncludeStyle::IBS_Regroup;
  EXPECT_EQ("#include <a>\n"
            "#include <b>\n"
            "#include <c>",
            sort("#include <b>\n"
                 "#include <a>\n"
                 "\n"
                 "#include <b>\n"
                 "\n"
                 "#include <c>\n"
                 "#include <b>"));
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
            "#include <c>",
            sort("#include <a>\n"
                 "#include <b>\n"
                 "\n"
                 "#include <c>\n"
                 "#include <b>\n"
                 "#include <b>"));
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
                 "-->",
                 "input.h", 0));
}

TEST_F(SortIncludesTest, DoNotOutputReplacementsForSortedBlocksWithRegrouping) {
  Style.IncludeBlocks = Style.IBS_Regroup;
  std::string Code = R"(
#include "b.h"

#include <a.h>
)";
  EXPECT_EQ(Code, sort(Code, "input.h", 0));
}

TEST_F(SortIncludesTest,
       DoNotOutputReplacementsForSortedBlocksWithRegroupingWindows) {
  Style.IncludeBlocks = Style.IBS_Regroup;
  std::string Code = "#include \"b.h\"\r\n"
                     "\r\n"
                     "#include <a.h>\r\n";
  EXPECT_EQ(Code, sort(Code, "input.h", 0));
}

TEST_F(SortIncludesTest, DoNotRegroupGroupsInGoogleObjCStyle) {
  FmtStyle = getGoogleStyle(FormatStyle::LK_ObjC);

  EXPECT_EQ("#include <a.h>\n"
            "#include <b.h>\n"
            "#include \"a.h\"",
            sort("#include <b.h>\n"
                 "#include <a.h>\n"
                 "#include \"a.h\""));
}

TEST_F(SortIncludesTest, DoNotTreatPrecompiledHeadersAsFirstBlock) {
  Style.IncludeBlocks = Style.IBS_Merge;
  std::string Code = "#include \"d.h\"\r\n"
                     "#include \"b.h\"\r\n"
                     "#pragma hdrstop\r\n"
                     "\r\n"
                     "#include \"c.h\"\r\n"
                     "#include \"a.h\"\r\n"
                     "#include \"e.h\"\r\n";

  std::string Expected = "#include \"b.h\"\r\n"
                         "#include \"d.h\"\r\n"
                         "#pragma hdrstop\r\n"
                         "\r\n"
                         "#include \"e.h\"\r\n"
                         "#include \"a.h\"\r\n"
                         "#include \"c.h\"\r\n";

  EXPECT_EQ(Expected, sort(Code, "e.cpp", 2));

  Code = "#include \"d.h\"\n"
         "#include \"b.h\"\n"
         "#pragma hdrstop( \"c:\\projects\\include\\myinc.pch\" )\n"
         "\n"
         "#include \"c.h\"\n"
         "#include \"a.h\"\n"
         "#include \"e.h\"\n";

  Expected = "#include \"b.h\"\n"
             "#include \"d.h\"\n"
             "#pragma hdrstop(\"c:\\projects\\include\\myinc.pch\")\n"
             "\n"
             "#include \"e.h\"\n"
             "#include \"a.h\"\n"
             "#include \"c.h\"\n";

  EXPECT_EQ(Expected, sort(Code, "e.cpp", 2));
}

TEST_F(SortIncludesTest, skipUTF8ByteOrderMarkMerge) {
  Style.IncludeBlocks = Style.IBS_Merge;
  std::string Code = "\xEF\xBB\xBF#include \"d.h\"\r\n"
                     "#include \"b.h\"\r\n"
                     "\r\n"
                     "#include \"c.h\"\r\n"
                     "#include \"a.h\"\r\n"
                     "#include \"e.h\"\r\n";

  std::string Expected = "\xEF\xBB\xBF#include \"e.h\"\r\n"
                         "#include \"a.h\"\r\n"
                         "#include \"b.h\"\r\n"
                         "#include \"c.h\"\r\n"
                         "#include \"d.h\"\r\n";

  EXPECT_EQ(Expected, sort(Code, "e.cpp", 1));
}

TEST_F(SortIncludesTest, skipUTF8ByteOrderMarkPreserve) {
  Style.IncludeBlocks = Style.IBS_Preserve;
  std::string Code = "\xEF\xBB\xBF#include \"d.h\"\r\n"
                     "#include \"b.h\"\r\n"
                     "\r\n"
                     "#include \"c.h\"\r\n"
                     "#include \"a.h\"\r\n"
                     "#include \"e.h\"\r\n";

  std::string Expected = "\xEF\xBB\xBF#include \"b.h\"\r\n"
                         "#include \"d.h\"\r\n"
                         "\r\n"
                         "#include \"a.h\"\r\n"
                         "#include \"c.h\"\r\n"
                         "#include \"e.h\"\r\n";

  EXPECT_EQ(Expected, sort(Code, "e.cpp", 2));
}

TEST_F(SortIncludesTest, MergeLines) {
  Style.IncludeBlocks = Style.IBS_Merge;
  std::string Code = "#include \"c.h\"\r\n"
                     "#include \"b\\\r\n"
                     ".h\"\r\n"
                     "#include \"a.h\"\r\n";

  std::string Expected = "#include \"a.h\"\r\n"
                         "#include \"b\\\r\n"
                         ".h\"\r\n"
                         "#include \"c.h\"\r\n";

  EXPECT_EQ(Expected, sort(Code, "a.cpp", 1));
}

TEST_F(SortIncludesTest, DisableFormatDisablesIncludeSorting) {
  StringRef Sorted = "#include <a.h>\n"
                     "#include <b.h>\n";
  StringRef Unsorted = "#include <b.h>\n"
                       "#include <a.h>\n";
  EXPECT_EQ(Sorted, sort(Unsorted));
  FmtStyle.DisableFormat = true;
  EXPECT_EQ(Unsorted, sort(Unsorted, "input.cpp", 0));
}

TEST_F(SortIncludesTest, DisableRawStringLiteralSorting) {

  EXPECT_EQ("const char *t = R\"(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")\";",
            sort("const char *t = R\"(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")\";",
                 "test.cxx", 0));
  EXPECT_EQ("const char *t = R\"x(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")x\";",
            sort("const char *t = R\"x(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")x\";",
                 "test.cxx", 0));
  EXPECT_EQ("const char *t = R\"xyz(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")xyz\";",
            sort("const char *t = R\"xyz(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")xyz\";",
                 "test.cxx", 0));

  EXPECT_EQ("#include <a.h>\n"
            "#include <b.h>\n"
            "const char *t = R\"(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")\";\n"
            "#include <c.h>\n"
            "#include <d.h>\n"
            "const char *t = R\"x(\n"
            "#include <f.h>\n"
            "#include <e.h>\n"
            ")x\";\n"
            "#include <g.h>\n"
            "#include <h.h>\n"
            "const char *t = R\"xyz(\n"
            "#include <j.h>\n"
            "#include <i.h>\n"
            ")xyz\";\n"
            "#include <k.h>\n"
            "#include <l.h>",
            sort("#include <b.h>\n"
                 "#include <a.h>\n"
                 "const char *t = R\"(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")\";\n"
                 "#include <d.h>\n"
                 "#include <c.h>\n"
                 "const char *t = R\"x(\n"
                 "#include <f.h>\n"
                 "#include <e.h>\n"
                 ")x\";\n"
                 "#include <h.h>\n"
                 "#include <g.h>\n"
                 "const char *t = R\"xyz(\n"
                 "#include <j.h>\n"
                 "#include <i.h>\n"
                 ")xyz\";\n"
                 "#include <l.h>\n"
                 "#include <k.h>",
                 "test.cc", 4));

  EXPECT_EQ("const char *t = R\"AMZ029amz(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")AMZ029amz\";",
            sort("const char *t = R\"AMZ029amz(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")AMZ029amz\";",
                 "test.cxx", 0));

  EXPECT_EQ("const char *t = R\"-AMZ029amz(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")-AMZ029amz\";",
            sort("const char *t = R\"-AMZ029amz(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")-AMZ029amz\";",
                 "test.cxx", 0));

  EXPECT_EQ("const char *t = R\"AMZ029amz-(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")AMZ029amz-\";",
            sort("const char *t = R\"AMZ029amz-(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")AMZ029amz-\";",
                 "test.cxx", 0));

  EXPECT_EQ("const char *t = R\"AM|029amz-(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")AM|029amz-\";",
            sort("const char *t = R\"AM|029amz-(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")AM|029amz-\";",
                 "test.cxx", 0));

  EXPECT_EQ("const char *t = R\"AM[029amz-(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")AM[029amz-\";",
            sort("const char *t = R\"AM[029amz-(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")AM[029amz-\";",
                 "test.cxx", 0));

  EXPECT_EQ("const char *t = R\"AM]029amz-(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")AM]029amz-\";",
            sort("const char *t = R\"AM]029amz-(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")AM]029amz-\";",
                 "test.cxx", 0));

#define X "AMZ029amz{}+!%*=_:;',.<>|/?#~-$"

  EXPECT_EQ("const char *t = R\"" X "(\n"
            "#include <b.h>\n"
            "#include <a.h>\n"
            ")" X "\";",
            sort("const char *t = R\"" X "(\n"
                 "#include <b.h>\n"
                 "#include <a.h>\n"
                 ")" X "\";",
                 "test.cxx", 0));

#undef X
}

} // end namespace
} // end namespace format
} // end namespace clang
