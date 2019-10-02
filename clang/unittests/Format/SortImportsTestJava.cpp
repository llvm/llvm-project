#include "clang/Format/Format.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace {

class SortImportsTestJava : public ::testing::Test {
protected:
  std::vector<tooling::Range> GetCodeRange(StringRef Code) {
    return std::vector<tooling::Range>(1, tooling::Range(0, Code.size()));
  }

  std::string sort(StringRef Code, std::vector<tooling::Range> Ranges) {
    auto Replaces = sortIncludes(FmtStyle, Code, Ranges, "input.java");
    Ranges = tooling::calculateRangesAfterReplacements(Replaces, Ranges);
    auto Sorted = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Sorted));
    auto Result = applyAllReplacements(
        *Sorted, reformat(FmtStyle, *Sorted, Ranges, "input.java"));
    EXPECT_TRUE(static_cast<bool>(Result));
    return *Result;
  }

  std::string sort(StringRef Code) { return sort(Code, GetCodeRange(Code)); }

  FormatStyle FmtStyle;

public:
  SortImportsTestJava() {
    FmtStyle = getGoogleStyle(FormatStyle::LK_Java);
    FmtStyle.JavaImportGroups = {"com.test", "org", "com"};
    FmtStyle.SortIncludes = true;
  }
};

TEST_F(SortImportsTestJava, StaticSplitFromNormal) {
  EXPECT_EQ("import static org.b;\n"
            "\n"
            "import org.a;\n",
            sort("import org.a;\n"
                 "import static org.b;\n"));
}

TEST_F(SortImportsTestJava, CapitalBeforeLowercase) {
  EXPECT_EQ("import org.Test;\n"
            "import org.a.Test;\n"
            "import org.b;\n",
            sort("import org.a.Test;\n"
                 "import org.Test;\n"
                 "import org.b;\n"));
}

TEST_F(SortImportsTestJava, KeepSplitGroupsWithOneNewImport) {
  EXPECT_EQ("import static com.test.a;\n"
            "\n"
            "import static org.a;\n"
            "\n"
            "import static com.a;\n"
            "\n"
            "import com.test.b;\n"
            "import com.test.c;\n"
            "\n"
            "import org.b;\n"
            "\n"
            "import com.b;\n",
            sort("import static com.test.a;\n"
                 "\n"
                 "import static org.a;\n"
                 "\n"
                 "import static com.a;\n"
                 "\n"
                 "import com.test.b;\n"
                 "\n"
                 "import org.b;\n"
                 "\n"
                 "import com.b;\n"
                 "import com.test.c;\n"));
}

TEST_F(SortImportsTestJava, SplitGroupsWithNewline) {
  EXPECT_EQ("import static com.test.a;\n"
            "\n"
            "import static org.a;\n"
            "\n"
            "import static com.a;\n"
            "\n"
            "import com.test.b;\n"
            "\n"
            "import org.b;\n"
            "\n"
            "import com.b;\n",
            sort("import static com.test.a;\n"
                 "import static org.a;\n"
                 "import static com.a;\n"
                 "import com.test.b;\n"
                 "import org.b;\n"
                 "import com.b;\n"));
}

TEST_F(SortImportsTestJava, UnspecifiedGroupAfterAllGroups) {
  EXPECT_EQ("import com.test.a;\n"
            "\n"
            "import org.a;\n"
            "\n"
            "import com.a;\n"
            "\n"
            "import abc.a;\n"
            "import xyz.b;\n",
            sort("import com.test.a;\n"
                 "import com.a;\n"
                 "import xyz.b;\n"
                 "import abc.a;\n"
                 "import org.a;\n"));
}

TEST_F(SortImportsTestJava, NoSortOutsideRange) {
  std::vector<tooling::Range> Ranges = {tooling::Range(27, 15)};
  EXPECT_EQ("import org.b;\n"
            "import org.a;\n"
            "// comments\n"
            "// that do\n"
            "// nothing\n",
            sort("import org.b;\n"
                 "import org.a;\n"
                 "// comments\n"
                 "// that do\n"
                 "// nothing\n",
                 Ranges));
}

TEST_F(SortImportsTestJava, SortWhenRangeContainsOneLine) {
  std::vector<tooling::Range> Ranges = {tooling::Range(27, 20)};
  EXPECT_EQ("import org.a;\n"
            "import org.b;\n"
            "\n"
            "import com.a;\n"
            "// comments\n"
            "// that do\n"
            "// nothing\n",
            sort("import org.b;\n"
                 "import org.a;\n"
                 "import com.a;\n"
                 "// comments\n"
                 "// that do\n"
                 "// nothing\n",
                 Ranges));
}

TEST_F(SortImportsTestJava, SortLexicographically) {
  EXPECT_EQ("import org.a.*;\n"
            "import org.a.a;\n"
            "import org.aA;\n"
            "import org.aa;\n",
            sort("import org.aa;\n"
                 "import org.a.a;\n"
                 "import org.a.*;\n"
                 "import org.aA;\n"));
}

TEST_F(SortImportsTestJava, StaticInCommentHasNoEffect) {
  EXPECT_EQ("import org.a; // static\n"
            "import org.b;\n"
            "import org.c; // static\n",
            sort("import org.a; // static\n"
                 "import org.c; // static\n"
                 "import org.b;\n"));
}

TEST_F(SortImportsTestJava, CommentsWithAffectedImports) {
  EXPECT_EQ("import org.a;\n"
            "// commentB\n"
            "/* commentB\n"
            " commentB*/\n"
            "import org.b;\n"
            "// commentC\n"
            "import org.c;\n",
            sort("import org.a;\n"
                 "// commentC\n"
                 "import org.c;\n"
                 "// commentB\n"
                 "/* commentB\n"
                 " commentB*/\n"
                 "import org.b;\n"));
}

TEST_F(SortImportsTestJava, CommentWithUnaffectedImports) {
  EXPECT_EQ("import org.a;\n"
            "// comment\n"
            "import org.b;\n",
            sort("import org.a;\n"
                 "// comment\n"
                 "import org.b;\n"));
}

TEST_F(SortImportsTestJava, CommentAfterAffectedImports) {
  EXPECT_EQ("import org.a;\n"
            "import org.b;\n"
            "// comment\n",
            sort("import org.b;\n"
                 "import org.a;\n"
                 "// comment\n"));
}

TEST_F(SortImportsTestJava, CommentBeforeAffectedImports) {
  EXPECT_EQ("// comment\n"
            "import org.a;\n"
            "import org.b;\n",
            sort("// comment\n"
                 "import org.b;\n"
                 "import org.a;\n"));
}

TEST_F(SortImportsTestJava, FormatTotallyOff) {
  EXPECT_EQ("// clang-format off\n"
            "import org.b;\n"
            "import org.a;\n"
            "// clang-format on\n",
            sort("// clang-format off\n"
                 "import org.b;\n"
                 "import org.a;\n"
                 "// clang-format on\n"));
}

TEST_F(SortImportsTestJava, FormatTotallyOn) {
  EXPECT_EQ("// clang-format off\n"
            "// clang-format on\n"
            "import org.a;\n"
            "import org.b;\n",
            sort("// clang-format off\n"
                 "// clang-format on\n"
                 "import org.b;\n"
                 "import org.a;\n"));
}

TEST_F(SortImportsTestJava, FormatPariallyOnShouldNotReorder) {
  EXPECT_EQ("// clang-format off\n"
            "import org.b;\n"
            "import org.a;\n"
            "// clang-format on\n"
            "import org.d;\n"
            "import org.c;\n",
            sort("// clang-format off\n"
                 "import org.b;\n"
                 "import org.a;\n"
                 "// clang-format on\n"
                 "import org.d;\n"
                 "import org.c;\n"));
}

TEST_F(SortImportsTestJava, DeduplicateImports) {
  EXPECT_EQ("import org.a;\n", sort("import org.a;\n"
                                    "import org.a;\n"));
}

TEST_F(SortImportsTestJava, NoNewlineAtEnd) {
  EXPECT_EQ("import org.a;\n"
            "import org.b;",
            sort("import org.b;\n"
                 "import org.a;"));
}

TEST_F(SortImportsTestJava, ImportNamedFunction) {
  EXPECT_EQ("import X;\n"
            "class C {\n"
            "  void m() {\n"
            "    importFile();\n"
            "  }\n"
            "}\n",
            sort("import X;\n"
                 "class C {\n"
                 "  void m() {\n"
                 "    importFile();\n"
                 "  }\n"
                 "}\n"));
}

TEST_F(SortImportsTestJava, NoReplacementsForValidImports) {
  // Identical #includes have led to a failure with an unstable sort.
  std::string Code = "import org.a;\n"
                     "import org.b;\n";
  EXPECT_TRUE(
      sortIncludes(FmtStyle, Code, GetCodeRange(Code), "input.java").empty());
}

TEST_F(SortImportsTestJava, NoReplacementsForValidImportsWindows) {
  std::string Code = "import org.a;\r\n"
                     "import org.b;\r\n";
  EXPECT_TRUE(
      sortIncludes(FmtStyle, Code, GetCodeRange(Code), "input.java").empty());
}

} // end namespace
} // end namespace format
} // end namespace clang
