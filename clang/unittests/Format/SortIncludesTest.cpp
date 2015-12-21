//===- unittest/Format/SortIncludesTest.cpp - Include sort unit tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  std::string sort(llvm::StringRef Code, StringRef FileName = "input.cpp") {
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    std::string Sorted =
        applyAllReplacements(Code, sortIncludes(Style, Code, Ranges, FileName));
    return applyAllReplacements(Sorted,
                                reformat(Style, Sorted, Ranges, FileName));
  }

  unsigned newCursor(llvm::StringRef Code, unsigned Cursor) {
    std::vector<tooling::Range> Ranges(1, tooling::Range(0, Code.size()));
    sortIncludes(Style, Code, Ranges, "input.cpp", &Cursor);
    return Cursor;
  }

  FormatStyle Style = getLLVMStyle();

};

TEST_F(SortIncludesTest, BasicSorting) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"a.h\"\n"
                 "#include \"c.h\"\n"
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
  Style.SortIncludes = false;
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

TEST_F(SortIncludesTest, HandlesAngledIncludesAsSeparateBlocks) {
  EXPECT_EQ("#include \"a.h\"\n"
            "#include \"c.h\"\n"
            "#include <b.h>\n"
            "#include <d.h>\n",
            sort("#include <d.h>\n"
                 "#include <b.h>\n"
                 "#include \"c.h\"\n"
                 "#include \"a.h\"\n"));

  Style = getGoogleStyle(FormatStyle::LK_Cpp);
  EXPECT_EQ("#include <b.h>\n"
            "#include <d.h>\n"
            "#include \"a.h\"\n"
            "#include \"c.h\"\n",
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
                 "a_main.cc"));
  EXPECT_EQ("#include \"llvm/input.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"llvm/input.h\"\n"
                 "#include \"c.h\"\n"
                 "#include \"b.h\"\n",
                 "input.mm"));

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

} // end namespace
} // end namespace format
} // end namespace clang
