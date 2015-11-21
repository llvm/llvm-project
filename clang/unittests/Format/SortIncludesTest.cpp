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
                 "#include \"b.h\"\n"));
  EXPECT_EQ("#include \"llvm/a.h\"\n"
            "#include \"b.h\"\n"
            "#include \"c.h\"\n",
            sort("#include \"llvm/a.h\"\n"
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
                 "some_header.h"));
}

} // end namespace
} // end namespace format
} // end namespace clang
